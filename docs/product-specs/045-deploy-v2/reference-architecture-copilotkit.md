# Reference architecture: CopilotKit app with Cloudflare Access

**Status:** Draft for review
**Date:** 2026-11-16
**Belongs to:** [045 Deploy v2](spec.md), decision D11 and requirements FR-C48 to FR-C56

This document shows one complete way to put user sign-in in front of a
HoloDeck agent on Cloudflare. It uses the CopilotKit sample
(`sample/copilotkit`) as the user interface. It is a reference, not the only
supported layout.

## 1. What it gives you

- Users sign in with an identity provider that you already have: Google,
  Microsoft Entra, GitHub, or a one-time code by email.
- Only the people in your Access policy reach the app.
- Only the app reaches the agent. A browser never calls the agent directly
  and never holds a credential for it.
- Each user sees only their own conversations.

It does not give you public sign-up. Cloudflare Access has no user store. For
open registration, use a separate identity product in front of the app and
keep the service-token hop in §3.

## 2. Diagram

```
                         identity provider (Google / Entra / GitHub)
                                   ▲
                                   │ sign-in
browser ──▶ Access application A ──┴─▶ CopilotKit app (Next.js on Workers)
            policy: your users          │ reads the user from the Access token
                                        │ builds a thread ID that belongs to the user
                                        │ adds the service token headers
                                        ▼
            Access application B ─────▶ HoloDeck Worker ─▶ Durable Object ─▶ container
            policy: Service Auth,        verifies the Access token (FR-C48)
            this one service token       records who owns each thread (FR-C52)
```

## 3. The two hops

**Hop 1: browser to the CopilotKit app.** Access application A covers the
app's URL. Its policy allows your users, by email domain, by address, or by
an identity-provider group. Access sends a user who is not signed in to the
identity provider, then passes a signed token to the app on each request. The
app has no login page.

**Hop 2: the app's server to the HoloDeck agent.** Access application B covers
the agent's Worker URL. Its policy has the action **Service Auth** and allows
one service token. The app's server route sends that token in the
`CF-Access-Client-Id` and `CF-Access-Client-Secret` headers. Access admits the
request and passes a signed token to the Worker, which verifies it.

The service token replaces the static bearer token of token mode. It can be
rotated with a grace period and revoked on its own.

## 4. What each part must do

**The CopilotKit app**

- Verify the Access token on each request to its server routes. Do not trust
  the browser for the user's identity.
- Build each AG-UI thread ID on the server so that it contains the user's
  stable ID from the Access token. Refuse a thread ID from the browser that
  does not belong to the signed-in user.
- Hold the service token as a server-side secret. It must not appear in
  client code or in a response.
- Set `HOLODECK_BACKEND_URL` to the agent's `/awp` URL, as the sample does today.

**The HoloDeck deployment**

```yaml
deployment:
  target:
    provider: cloudflare
    cloudflare:
      account_id: ${CLOUDFLARE_ACCOUNT_ID}
      secrets: [ANTHROPIC_API_KEY]
      auth:
        access:
          team_domain: https://example-team.cloudflareaccess.com
          audience: <AUD tag of Access application B>
```

The Worker verifies the token, passes the caller's identity to the container,
and refuses a thread that another caller owns (FR-C48 to FR-C52).

## 5. Who the agent sees

In this layout the agent's caller is the app, not the person. The Worker's
ownership check (FR-C52) therefore separates one calling service from
another. It does not separate one user from another: that is the app's job,
through the thread ID rule in §4.

A layout where the agent sees each person needs the user's own Access token
to reach the agent. That is not part of this reference. It is open question 7
in the spec.

## 6. Setup order

1. Create the identity provider connection in Cloudflare Zero Trust.
2. Deploy the agent with `holodeck deploy build` and `holodeck deploy run`.
3. Create Access application B on the agent's URL, with a Service Auth policy
   and a new service token. Put its AUD tag in the agent config and deploy
   again. `deploy run` reports an error while no Access application covers
   the URL (FR-C54).
4. Deploy the CopilotKit app with the service token as two secrets.
5. Create Access application A on the app's URL, with a policy for your users.

## 7. Checks

- A browser that is not signed in is sent to the identity provider.
- A signed-in user who is not in the policy is refused by Access.
- `curl` to the agent's URL without the service token never reaches a container.
- User A cannot open user B's thread by sending its ID.
- After the service token is rotated, the app works again with the new secret,
  and the old secret stops working when its grace period ends.

## 8. Not verified yet

This layout has not been built. The spec's open questions 5 to 7 list the
platform facts to confirm, and SC-10 is its acceptance test.
