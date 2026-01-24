# Feature Specification: HoloDeck Deploy Command

**Feature Branch**: `019-deploy-command`
**Created**: 2026-01-24
**Status**: Draft
**Input**: User description: "Build a deployment pipeline for HoloDeck agents that enables developers to containerize and deploy their agents to cloud platforms through a single CLI command."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build Container Image for Agent (Priority: P1)

As a developer, I want to build a container image from my agent configuration so that I can package my agent with all its dependencies for deployment.

**Why this priority**: This is the foundational capability that enables all subsequent deployment steps. Without a built image, nothing else can happen.

**Independent Test**: Can be fully tested by running the build command on a valid agent.yaml and verifying a container image is produced locally. Delivers value by enabling local testing of containerized agents.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml with instructions and data files, **When** I run `holodeck deploy build`, **Then** a container image is created locally with the agent name as the image tag
2. **Given** an agent configuration referencing external files (instructions, data), **When** I run `holodeck deploy build`, **Then** all referenced files are copied into the image
3. **Given** no container runtime is installed, **When** I run `holodeck deploy build`, **Then** a clear error message indicates which runtimes are supported and how to install them
4. **Given** the build command with `--dry-run` flag, **When** I execute it, **Then** I see what would be built without actually building

---

### User Story 2 - Push Image to Container Registry (Priority: P2)

As a developer, I want to push my built agent image to a container registry so that it can be accessed by cloud deployment platforms.

**Why this priority**: Registry push is required before cloud deployment but depends on having a built image first.

**Independent Test**: Can be fully tested by pushing an image to a registry and verifying it appears in the registry. Delivers value by enabling image distribution and versioning.

**Acceptance Scenarios**:

1. **Given** a locally built image and valid registry credentials, **When** I run `holodeck deploy push`, **Then** the image is pushed to the configured registry with the specified tag
2. **Given** registry configuration in config.yaml with tag strategy set to `git-sha`, **When** I run `holodeck deploy push`, **Then** the image is tagged with the current git commit SHA
3. **Given** invalid or missing registry credentials, **When** I run `holodeck deploy push`, **Then** a clear error message explains the authentication failure and how to configure credentials
4. **Given** no locally built image exists, **When** I run `holodeck deploy push`, **Then** a clear error message instructs me to run `holodeck deploy build` first

---

### User Story 3 - Deploy Agent to Cloud Platform (Priority: P3)

As a developer, I want to deploy my containerized agent to a cloud platform so that it becomes accessible as an API endpoint.

**Why this priority**: This is the final step that delivers user value by making the agent accessible, but depends on both build and push completing first.

**Independent Test**: Can be fully tested by deploying to a cloud platform and verifying the agent responds to HTTP requests. Delivers value by providing a production-ready API endpoint.

**Acceptance Scenarios**:

1. **Given** an image in a registry and valid cloud credentials for Azure, **When** I run `holodeck deploy run`, **Then** the agent is deployed to Azure Container Apps and the deployment URL is displayed
2. **Given** an image in a registry and valid cloud credentials for Google Cloud, **When** I run `holodeck deploy run`, **Then** the agent is deployed to Cloud Run and the deployment URL is displayed
3. **Given** an image in a registry and valid cloud credentials for AWS, **When** I run `holodeck deploy run`, **Then** the agent is deployed to App Runner and the deployment URL is displayed
4. **Given** environment variables configured for secrets (API keys), **When** I deploy, **Then** those secrets are injected into the running container without being stored in the image
5. **Given** the image does not exist in the registry, **When** I run `holodeck deploy run`, **Then** a clear error message instructs me to run `holodeck deploy push` first

---

### User Story 4 - Full Pipeline Deployment (Priority: P4)

As a developer, I want to run a single command that builds, pushes, and deploys my agent so that I can go from code to production quickly.

**Why this priority**: Convenience feature that combines the previous three stories for streamlined workflow.

**Independent Test**: Can be fully tested by running the full deploy command and verifying the agent is accessible at the deployment URL. Delivers value by reducing deployment to a single command.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml and all required credentials, **When** I run `holodeck deploy` (no subcommand), **Then** the system builds, pushes, and deploys in sequence, showing progress for each step
2. **Given** the build step fails, **When** running the full pipeline, **Then** subsequent steps are skipped and a clear error indicates the failure point
3. **Given** `--dry-run` flag on full pipeline, **When** I execute it, **Then** I see the complete plan for all three steps without executing any of them

---

### Edge Cases

- What happens when the agent.yaml references files that don't exist? System should fail fast with a clear error listing missing files.
- How does the system handle interrupted builds or pushes? Partial artifacts should be cleaned up, and re-running should work from a clean state.
- What happens when the cloud platform quota is exceeded? Clear error with guidance on how to increase quotas or choose alternative regions.
- How does the system handle network failures during push? Retry logic with exponential backoff, eventually failing with actionable error message.
- What happens when deploying an agent that's already deployed? Update the existing deployment rather than creating a duplicate.

## Requirements *(mandatory)*

### Functional Requirements

**Container Image Building**
- **FR-001**: System MUST support building container images using Docker, Podman, or nerdctl (auto-detected or user-specified via `runtime` configuration)
- **FR-002**: System MUST use a HoloDeck base image that includes Python 3.10+ runtime and the holodeck package pre-installed
- **FR-003**: System MUST copy agent.yaml, instruction files, data directories, and all referenced artifacts into the container image
- **FR-004**: System MUST generate a Dockerfile dynamically based on agent configuration (users do not write Dockerfiles)
- **FR-005**: System MUST support build-time arguments and labels for versioning and metadata (image version, build timestamp, git SHA)

**Container Registry Push**
- **FR-006**: System MUST read registry configuration (URL, repository, credentials) from the deployment section of config.yaml
- **FR-007**: System MUST support these registries: Docker Hub, GitHub Container Registry (ghcr.io), Azure Container Registry, Google Artifact Registry, and AWS ECR
- **FR-008**: System MUST authenticate to registries via environment variables or native credential helpers
- **FR-009**: System MUST support tagging strategies: semantic versioning, git SHA, timestamp, or custom user-defined tags

**Cloud Deployment**
- **FR-010**: System MUST deploy to Azure Container Apps with configurable ingress (external/internal), auto-scaling (min/max replicas), and resource allocation (CPU/memory)
- **FR-011**: System MUST deploy to Google Cloud Run with configurable concurrency, memory allocation, and timeout settings
- **FR-012**: System MUST deploy to AWS App Runner with configurable auto-scaling and VPC configuration
- **FR-013**: System MUST inject environment variables for secrets (API keys, etc.) from configuration without baking them into the image
- **FR-014**: System MUST output the deployment URL and health status upon successful deployment
- **FR-027**: System MUST use Python SDKs (boto3 for AWS, google-cloud-run for GCP, azure-mgmt-containerinstance for Azure) as optional dependencies for cloud deployment, avoiding external CLI requirements

**CLI Interface**
- **FR-015**: System MUST provide `holodeck deploy build` subcommand to build container images only
- **FR-016**: System MUST provide `holodeck deploy push` subcommand to push images to registry (fails if no image built)
- **FR-017**: System MUST provide `holodeck deploy run` subcommand to deploy to cloud (fails if image not in registry)
- **FR-018**: System MUST run full pipeline (build, push, deploy) when `holodeck deploy` is invoked without a subcommand
- **FR-019**: System MUST support `--dry-run` flag on all commands to show planned actions without executing
- **FR-029**: System MUST provide `holodeck deploy status` subcommand to display deployment health, URL, and resource utilization for a deployed agent
- **FR-030**: System MUST provide `holodeck deploy destroy` subcommand to tear down a deployed agent and clean up cloud resources

**HoloDeck Base Image**
- **FR-020**: System MUST provide a base image with minimal Python 3.10+ runtime and UV package manager
- **FR-021**: Base image MUST include holodeck package pre-installed with `holodeck serve` as the container entrypoint command
- **FR-022**: Base image MUST serve the agent via HTTP using the existing `holodeck serve` command, exposing health check at `/health`, readiness at `/ready`, and OpenAPI docs at `/docs`
- **FR-023**: Base image MUST expose the agent API on a configurable port (default: 8080) with support for both REST protocol (`/agent/{name}/chat`) and AG-UI protocol (`/awp`)
- **FR-028**: System MUST allow users to configure which protocol(s) the deployed container exposes (REST, AG-UI, or both) via the deployment section of config.yaml, defaulting to REST

**Operational Requirements**
- **FR-024**: System MUST show progress indicators for all long-running operations (build, push, deploy)
- **FR-025**: System MUST provide detailed error messages with remediation hints for all failure scenarios
- **FR-026**: System MUST support non-interactive mode for CI/CD integration (no prompts, exit codes for success/failure)

### Key Entities

- **DeploymentConfig**: Represents the deployment section of config.yaml containing runtime preferences, registry settings, and cloud target configuration
- **ContainerImage**: Represents a built image with its name, tag, and local image ID
- **Registry**: Represents a container registry with URL, repository name, and authentication method
- **CloudDeployment**: Represents a deployed instance with provider, URL, status, and resource configuration
- **BaseImage**: Represents the HoloDeck base image used as the foundation for agent containers

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developers can go from agent.yaml to deployed API endpoint in under 10 minutes (excluding cloud provisioning time)
- **SC-002**: 95% of deployment attempts succeed on first try when all prerequisites (credentials, quotas) are properly configured
- **SC-003**: Developers can understand and resolve any deployment error within 2 minutes using the provided error message and remediation hints
- **SC-004**: The deployed agent health check endpoint responds within 30 seconds of deployment completion
- **SC-005**: Container image builds complete within 5 minutes for typical agent configurations (< 100MB of data files)
- **SC-006**: Deployment workflow requires zero custom code - all configuration via YAML only
- **SC-007**: CI/CD integration requires no interactive prompts and provides clear exit codes (0 = success, non-zero = failure with specific codes)

## Assumptions

- Users have appropriate cloud provider accounts and credentials configured before attempting deployment
- Users have a container runtime (Docker, Podman, or nerdctl) installed on their development machine
- The HoloDeck base image will be published to a public registry (Docker Hub) for easy access
- Network connectivity to container registries and cloud providers is available during deployment
- Users understand basic container and cloud deployment concepts

## Clarifications

### Session 2026-01-24

- Q: What API contract should deployed agents expose? → A: Use existing `holodeck serve` command which provides REST protocol (`/agent/{name}/chat`, `/agent/{name}/chat/stream`), AG-UI protocol (`/awp`), health checks (`/health`, `/ready`), and OpenAPI docs (`/docs`)
- Q: Should cloud deployment use CLIs or SDKs? → A: Python SDKs (boto3, google-cloud-run, azure-mgmt) bundled as optional dependencies
- Q: Which serve protocol should deployed containers expose? → A: Configurable via deployment config in YAML (REST, AG-UI, or both)
- Q: Should deploy command include lifecycle management? → A: Minimal set with `status` and `destroy` subcommands only; logs accessed via cloud console

## Out of Scope (v1)

- Kubernetes deployment (planned for future release)
- Multi-agent deployment in a single container
- Custom Dockerfile support (users must use the generated Dockerfile based on base image)
- Blue/green or canary deployment strategies
- GPU-enabled base images (CPU-only for v1)
- Automatic SSL certificate provisioning (relies on cloud provider defaults)
- Custom domain configuration
