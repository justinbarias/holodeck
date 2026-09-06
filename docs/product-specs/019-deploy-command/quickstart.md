# Quickstart: HoloDeck Deploy Command

This guide walks you through deploying a HoloDeck agent to the cloud.

## Prerequisites

1. **Docker installed** - [Get Docker](https://docs.docker.com/get-docker/)
2. **UV installed** - [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
3. **HoloDeck installed** - `uv pip install holodeck`
4. **Cloud provider SDK** (install one):
   - AWS: `uv pip install "holodeck[deploy-aws]"`
   - GCP: `uv pip install "holodeck[deploy-gcp]"`
   - Azure: `uv pip install "holodeck[deploy-azure]"`
   - All: `uv pip install "holodeck[deploy-all]"`

5. **Cloud credentials configured** (see provider-specific setup below)

---

## Quick Deploy (Full Pipeline)

```bash
# Deploy your agent with a single command
holodeck deploy agent.yaml
```

This command:
1. Builds a container image from your agent configuration
2. Pushes the image to your configured registry
3. Deploys to your cloud provider
4. Outputs the deployment URL

---

## Step-by-Step Deployment

### Step 1: Add Deployment Config to agent.yaml

Add a `deployment:` section to your existing `agent.yaml`:

```yaml
name: my-support-agent
model:
  provider: openai
  name: gpt-4o

instructions:
  file: instructions.md

# Add deployment configuration
deployment:
  registry:
    url: ghcr.io
    repository: my-org/agents
    tag_strategy: git-sha
  target:
    provider: gcp
    gcp:
      project_id: my-gcp-project
      region: us-central1
  environment:
    OPENAI_API_KEY: ${OPENAI_API_KEY}
```

### Step 2: Build the Container Image

```bash
holodeck deploy build agent.yaml
```

Output:
```
Building container image...
  Step 1/8: FROM holodeck/base:latest
  Step 2/8: COPY agent.yaml /app/
  ...
Successfully built image: ghcr.io/my-org/agents:abc12345
```

### Step 3: Push to Registry

```bash
holodeck deploy push agent.yaml
```

Output:
```
Pushing to ghcr.io...
  abc12345: Pushed
  latest: Pushed
Successfully pushed: ghcr.io/my-org/agents:abc12345
```

### Step 4: Deploy to Cloud

```bash
holodeck deploy run agent.yaml
```

Output:
```
Deploying to Google Cloud Run...
  Creating service: my-support-agent
  Waiting for deployment...
  Service is ready!

Deployment URL: https://my-support-agent-abc123-uc.a.run.app
Health check: https://my-support-agent-abc123-uc.a.run.app/health
```

---

## Check Deployment Status

```bash
holodeck deploy status agent.yaml
```

Output:
```
Service: my-support-agent
Provider: gcp (Cloud Run)
Status: RUNNING
URL: https://my-support-agent-abc123-uc.a.run.app
Last updated: 2026-01-24 10:35:00 UTC
```

---

## Destroy Deployment

```bash
holodeck deploy destroy agent.yaml
```

Output:
```
Destroying deployment: my-support-agent
  Deleting Cloud Run service...
  Service deleted.
Deployment destroyed successfully.
```

---

## Cloud Provider Setup

### AWS App Runner

**1. Create ECR Repository:**
```bash
aws ecr create-repository --repository-name my-agents
```

**2. Create IAM Role for App Runner:**
```bash
aws iam create-role \
  --role-name AppRunnerECRRole \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
  --role-name AppRunnerECRRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
```

**3. Configure agent.yaml:**
```yaml
deployment:
  registry:
    url: 123456789012.dkr.ecr.us-east-1.amazonaws.com
    repository: my-agents
  target:
    provider: aws
    aws:
      region: us-east-1
      ecr_role_arn: arn:aws:iam::123456789012:role/AppRunnerECRRole
      cpu: "1 vCPU"
      memory: "2 GB"
```

**4. Set credentials:**
```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1
```

---

### Google Cloud Run

**1. Enable APIs:**
```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

**2. Create Artifact Registry repository:**
```bash
gcloud artifacts repositories create agents \
  --repository-format=docker \
  --location=us-central1
```

**3. Configure agent.yaml:**
```yaml
deployment:
  registry:
    url: us-central1-docker.pkg.dev
    repository: my-project/agents/my-agent
  target:
    provider: gcp
    gcp:
      project_id: my-gcp-project
      region: us-central1
      memory: 512Mi
      concurrency: 80
```

**4. Set credentials:**
```bash
gcloud auth application-default login
# Or use service account:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

### Azure Container Apps

**1. Create resource group and Container Apps environment:**
```bash
az group create --name holodeck-rg --location eastus

az containerapp env create \
  --name holodeck-env \
  --resource-group holodeck-rg \
  --location eastus
```

**2. Create Azure Container Registry:**
```bash
az acr create --name myacr --resource-group holodeck-rg --sku Basic
az acr login --name myacr
```

**3. Configure agent.yaml:**
```yaml
deployment:
  registry:
    url: myacr.azurecr.io
    repository: holodeck-agents
  target:
    provider: azure
    azure:
      subscription_id: 12345678-1234-1234-1234-123456789012
      resource_group: holodeck-rg
      environment_name: holodeck-env
      location: eastus
      cpu: "0.5"
      memory: 1Gi
```

**4. Set credentials:**
```bash
az login
# Or use service principal:
export AZURE_TENANT_ID=your-tenant
export AZURE_CLIENT_ID=your-client-id
export AZURE_CLIENT_SECRET=your-secret
```

---

## Common Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be done without executing |
| `--verbose, -v` | Enable detailed output |
| `--quiet, -q` | Suppress progress output (for CI/CD) |

```bash
# Preview deployment without executing
holodeck deploy --dry-run agent.yaml

# Verbose output for debugging
holodeck deploy -v agent.yaml

# Quiet mode for CI/CD pipelines
holodeck deploy -q agent.yaml
```

---

## Environment Variables

Pass secrets to your deployed agent via environment variables:

```yaml
deployment:
  environment:
    OPENAI_API_KEY: ${OPENAI_API_KEY}
    DATABASE_URL: ${DATABASE_URL}
    LOG_LEVEL: INFO
```

Variables with `${VAR}` syntax are resolved from your local environment at deploy time. The values are injected into the container at runtime (not baked into the image).

---

## Testing Your Deployment

Once deployed, test your agent:

```bash
# Health check
curl https://your-deployment-url/health

# Send a message (REST protocol)
curl -X POST https://your-deployment-url/agent/my-agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how can you help me?"}'
```

---

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Deploy Agent

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install UV
        uses: astral-sh/setup-uv@v4

      - name: Install HoloDeck
        run: uv pip install "holodeck[deploy-gcp]"

      - name: Deploy
        env:
          GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GCP_SA_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          echo '${{ secrets.GCP_SA_KEY }}' > /tmp/sa-key.json
          export GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa-key.json
          holodeck deploy -q agent.yaml
```

---

## Troubleshooting

### Docker not found
```
Error: Docker not available. Install Docker Desktop or Docker Engine.
```
**Solution**: Install Docker from https://docs.docker.com/get-docker/

### Cloud SDK not installed
```
Error: AWS SDK not installed. Run: uv pip install "holodeck[deploy-aws]"
```
**Solution**: Install the appropriate cloud SDK extra using `uv pip install "holodeck[deploy-<provider>]"`.

### Authentication failed
```
Error: Failed to authenticate with registry. Check your credentials.
```
**Solution**: Ensure cloud credentials are configured correctly (see provider setup above).

### Image push failed
```
Error: Push failed: unauthorized
```
**Solution**: Log in to your container registry:
- Docker Hub: `docker login`
- GHCR: `echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin`
- ECR: `aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ECR_URL`
- ACR: `az acr login --name YOUR_ACR_NAME`
- GAR: `gcloud auth configure-docker us-central1-docker.pkg.dev`
