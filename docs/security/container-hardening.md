# Container Hardening Posture (Spec 034 P2a)

When you `holodeck deploy run`, HoloDeck applies the following hardening to
the generated Container App. All defaults are on; no agent.yaml changes
are required.

| Layer | Default | Enforcement layer | How to override |
|---|---|---|---|
| Runtime user | UID 1000 (`holodeck`), non-root | Dockerfile `USER` directive | Cannot be raised to root in generated images. |
| Corpus filesystems (`/app/data`, `/app/instructions`) | Owned by root, read-only (chmod a-w) | Dockerfile | Move writable content to `/var/holodeck/work` via the tool/SDK. |
| Ephemeral scratch (`/var/holodeck/work`, `/tmp`) | ACA EmptyDir volume per replica | ACA `Volume`/`VolumeMount` | Size scales with the replica's memory limit; cleared on replica restart. |
| Container ingress | Internal (`ingress_external: false`) | ACA Ingress | Set `deployment.target.azure.ingress_external: true` (warning emitted). |
| Node.js | Installed only when an MCP server's `command` starts with `node`/`npx`/`yarn`/`pnpm` | Generated Dockerfile | Add a stdio MCP server with one of those commands. |
| `bubblewrap` (`bwrap`) | Always installed in base image | `docker/Dockerfile` | Required by the SDK for `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`. Set `claude.disable_subprocess_env_scrub: true` to skip (warning emitted). |
| Credential-shaped files in COPY surface | Warned at `deploy build` (filenames only; no content scan) | CLI lint | Remove the file, or accept the warning if the file is legitimately public. |

### Posture ACA does NOT enforce (and we don't claim to)

The Anthropic secure-deployment guide recommends `--cap-drop ALL`,
`--security-opt no-new-privileges`, `--read-only` root FS, and seccomp
profiles. **ACA does not expose any of these primitives** at any API
version (verified 2026-05-23 against `azure-mgmt-appcontainers==4.0.0`
and ARM API 2026-01-01). The ACA platform default cap set / seccomp
profile is Microsoft-controlled and not customer-tunable.

The image-layer enforcement points (non-root user, read-only corpus,
ephemeral scratch) are the closest *equivalents* HoloDeck can produce
on ACA. For threat models that require the strict Kubernetes
securityContext primitives, see [`aca-limitations.md`](aca-limitations.md)
for the AKS escape hatch.

## What hardening does NOT cover

The following recommendations from
[Anthropic's secure-deployment guide](https://code.claude.com/docs/en/agent-sdk/secure-deployment)
are outside P2a:

- **`cap-drop ALL` / `no-new-privileges` / `readOnlyRootFilesystem` / seccomp.**
  Not expressible in the ACA management surface. See
  [`aca-limitations.md`](aca-limitations.md).
- **Network egress restrictions.** Default profile allows the container to
  reach any HTTPS endpoint. The `hardened` security profile (spec 034 P3,
  not yet shipped) is the way to restrict egress.
- **Credential boundary (proxy pattern).** Credentials remain in container
  env vars in the default profile. Move to `hardened` for the Envoy
  credential-injector pattern.
- **`--userns-remap`, `--ipc private`, `--pids-limit`.** Not exposed by ACA.
- **gVisor / Firecracker / VM isolation.** ACA cannot host these.

## Verifying hardening is in effect

```bash
# After `holodeck deploy run`, check the deploy-time echo for the
# ACA-enforced + image-enforced lines:
#   Ephemeral scratch (ACA EmptyDir): /tmp, /var/holodeck/work
#   Image-layer hardening: non-root (UID 1000), corpus read-only (/app/data, /app/instructions)
#   Note: ACA security context primitives ...

# Confirm runtime non-root + read-only corpus on a deployed agent:
az containerapp exec --resource-group <rg> --name <agent> \
    --command "sh -c 'id -u; test ! -w /app/data && echo READONLY_OK'"
# Expect: 1000 \n READONLY_OK
```
