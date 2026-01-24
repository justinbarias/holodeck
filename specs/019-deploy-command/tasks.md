# Tasks: HoloDeck Deploy Command

**Input**: Design documents from `/specs/019-deploy-command/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/deployment-config-schema.yaml, quickstart.md
**Approach**: TDD (Test-Driven Development) - Tests written first, must fail before implementation

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Reference Locations

Use these references when implementing each task:

| Document | Path | Key Sections |
|----------|------|--------------|
| Spec | @spec.md | User Stories (L10-75), Requirements (L86-131) |
| Plan | @plan.md | Project Structure (L69-126), Component Design (L130-287) |
| Schema | @contracts/deployment-config-schema.yaml | Full schema (L1-342) |
| Data Model | @data-model.md | Entity definitions (L101-285) |
| Research | @research.md | SDK patterns (L26-88, L151-437) |
| Quickstart | @quickstart.md | Usage examples (L20-109) |

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency configuration

- [x] T001 Add optional deployment dependencies to pyproject.toml extras
  - Spec details: @plan.md L:17-20 (dependencies)
  - Add `deploy`, `deploy-aws`, `deploy-gcp`, `deploy-azure`, `deploy-all` extras
  - Packages: docker>=7.0.0, boto3>=1.42.0, google-cloud-run>=0.13.0, azure-mgmt-appcontainers>=4.0.0, azure-identity>=1.15.0

- [x] T002 [P] Create deploy package structure in src/holodeck/deploy/
  - Spec details: @plan.md L:85-102 (project structure)
  - Create directories: deploy/, deploy/deployers/
  - Create __init__.py files for each

- [x] T003 [P] Add DeploymentError exception to src/holodeck/lib/errors.py
  - Spec details: @plan.md L:106 (error handling)
  - Add DeploymentError class extending HoloDeckError

- [x] T004 [P] Create docker/ directory with base image files at repository root
  - Spec details: @plan.md L:107-111 (docker files)
  - Create docker/Dockerfile and docker/entrypoint.sh placeholders

- [x] T005 [P] Create test fixture directories in tests/
  - Spec details: @plan.md L:112-126 (test structure)
  - Create tests/unit/deploy/, tests/integration/deploy/, tests/fixtures/deploy/sample_agent/, tests/fixtures/deploy/mock_responses/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core models and utilities that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational Phase

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T006 [P] Unit tests for Pydantic deployment models in tests/unit/deploy/test_models.py
  - Schema details: @contracts/deployment-config-schema.yaml L:1-342
  - Data model details: @data-model.md L:103-235
  - Test DeploymentConfig, RegistryConfig, CloudTargetConfig, provider-specific configs
  - Test validation rules (patterns, enums, defaults)
  - Test discriminated union behavior for CloudTargetConfig

- [x] T007 [P] Unit tests for Dockerfile generation in tests/unit/deploy/test_dockerfile.py
  - Research details: @research.md L:92-128 (Dockerfile template)
  - Test Jinja2 template rendering
  - Test OCI label generation
  - Test file copy instructions for instructions and data directories

### Implementation for Foundational Phase

- [x] T008 Implement Pydantic models in src/holodeck/models/deployment.py
  - Schema details: @contracts/deployment-config-schema.yaml L:48-291 (definitions)
  - Data model details: @data-model.md L:103-235 (entity definitions)
  - Implement: RegistryConfig, AWSAppRunnerConfig, GCPCloudRunConfig, AzureContainerAppsConfig
  - Implement: CloudTargetConfig (discriminated union), DeploymentConfig
  - Add validators for all patterns and constraints

- [x] T009 Implement Dockerfile generator in src/holodeck/deploy/dockerfile.py
  - Research details: @research.md L:92-128 (Jinja2 template)
  - Create HOLODECK_DOCKERFILE_TEMPLATE constant
  - Implement generate_dockerfile() function accepting agent config
  - Support instruction files, data directories, environment variables

- [x] T010 Create HoloDeck base image Dockerfile in docker/Dockerfile
  - Spec details: @plan.md L:255-287 (base image definition)
  - Research details: @research.md L:535-538 (base image dependencies)
  - FROM python:3.10-slim, install UV, create non-root user
  - Install holodeck package, configure healthcheck
  - Set entrypoint to holodeck serve

- [x] T011 Create container entrypoint script in docker/entrypoint.sh
  - Spec details: @plan.md L:108 (entrypoint)
  - Handle environment variable configuration
  - Execute holodeck serve with appropriate flags

- [x] T012 Implement deployment config resolver in src/holodeck/deploy/config.py
  - Data model details: @data-model.md L:103-119 (DeploymentConfig)
  - Load deployment section from agent.yaml
  - Resolve environment variable substitutions (${VAR} syntax)
  - Validate against Pydantic models

**Checkpoint**: Foundation ready - Pydantic models, Dockerfile generation, and base image defined ✅

---

## Phase 2b: Build & Publish Base Image (Blocking Prerequisite)

**Purpose**: Build and publish the HoloDeck base image to a container registry so user agent images can use it as their `FROM` layer.

**⚠️ CRITICAL**: User Story 1 (build) requires the base image to exist in a registry. This phase must complete before Phase 3 can be fully tested.

### Implementation for Base Image Publishing

- [x] T012a [P] Create GitHub Actions workflow for base image in .github/workflows/build-base-image.yml
  - Trigger on: push to main (docker/ changes), manual dispatch, release tags
  - Build multi-arch image (linux/amd64, linux/arm64)
  - Push to GitHub Container Registry (ghcr.io/holodeck-ai/holodeck-base)
  - Tag with: latest, git-sha, semver (on release)
  - Use Docker Buildx for multi-platform builds

- [x] T012b [P] Add base image build script in scripts/build-base-image.sh
  - Local development script to build base image
  - Support --push flag for publishing
  - Support --tag flag for custom tags
  - Validate Dockerfile syntax before build

- [x] T012c Test base image locally before publishing
  - Build base image from docker/Dockerfile
  - Run container and verify `holodeck serve` starts
  - Verify healthcheck endpoint responds
  - Test with sample agent.yaml mounted

- [x] T012d Publish initial base image to GitHub Container Registry
  - Create ghcr.io/holodeck-ai/holodeck-base:latest
  - Create ghcr.io/holodeck-ai/holodeck-base:0.1.0 (initial version)
  - Update docker/Dockerfile generated by T009 to reference published image
  - Document image versioning strategy in docker/README.md

**Checkpoint**: Base image available at ghcr.io - agent builds can use `FROM ghcr.io/holodeck-ai/holodeck-base:latest` ✅

---

## Phase 3: User Story 1 - Build Container Image (Priority: P1) 🎯 MVP

**Goal**: Enable developers to build container images from agent.yaml using `holodeck deploy build`

**Independent Test**: Run `holodeck deploy build agent.yaml` and verify a container image is produced locally with correct tags and labels.

**Spec Reference**: @spec.md L:10-24 (User Story 1 acceptance scenarios)

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T013 [P] [US1] Unit tests for container builder in tests/unit/deploy/test_builder.py
  - Research details: @research.md L:26-88 (docker-py patterns)
  - Test Docker client initialization
  - Test build() method with streaming output
  - Test error handling for BuildError, DockerException
  - Mock docker.from_env() and images.build()

- [x] T014 [P] [US1] Integration test for image building in tests/integration/deploy/test_build_image.py
  - Spec details: @spec.md L:20-23 (acceptance scenarios)
  - Test with sample agent from tests/fixtures/deploy/sample_agent/
  - Test --dry-run flag behavior
  - Test error when Docker not available
  - Requires Docker daemon (mark with @pytest.mark.docker)

- [x] T015 [P] [US1] Create sample agent fixture in tests/fixtures/deploy/sample_agent/
  - Create minimal agent.yaml for testing
  - Create instructions.md file
  - Create sample data directory with test file

### Implementation for User Story 1

- [x] T016 [US1] Implement ContainerBuilder class in src/holodeck/deploy/builder.py
  - Research details: @research.md L:26-88 (docker-py SDK implementation)
  - Implement __init__() with docker.from_env()
  - Implement build() method with streaming logs
  - Implement tag generation (git-sha, semver, timestamp, custom)
  - Add OCI labels per @data-model.md L:251-256

- [x] T017 [US1] Implement `holodeck deploy build` CLI command in src/holodeck/cli/commands/deploy.py
  - Spec details: @spec.md L:112 (FR-015)
  - Plan details: @plan.md L:132-167 (CLI structure)
  - Create @click.group(name="deploy") with invoke_without_command=True
  - Implement build subcommand
  - Add --dry-run, --verbose, --quiet options
  - Show progress indicators for build steps
  - Output image name and tag on success

- [x] T018 [US1] Register deploy command in CLI main in src/holodeck/cli/main.py
  - Import and register deploy command group
  - Ensure deploy appears in holodeck --help

- [x] T019 [US1] Add validation for missing Docker runtime
  - Spec details: @spec.md L:22-23 (acceptance scenario 3)
  - Clear error message with installation instructions
  - Exit code 3 (execution error)

**Checkpoint**: `holodeck deploy build` works independently - can build images locally

---

## Phase 4: User Story 2 - Push Image to Registry (Priority: P2)

**Goal**: Enable developers to push built images to container registries using `holodeck deploy push`

**Independent Test**: Run `holodeck deploy push agent.yaml` and verify image appears in the configured registry.

**Spec Reference**: @spec.md L:27-41 (User Story 2 acceptance scenarios)

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T020 [P] [US2] Unit tests for registry pusher in tests/unit/deploy/test_pusher.py
  - Research details: @research.md L:70-88 (push implementation)
  - Test push() method with streaming output
  - Test authentication via auth_config
  - Test error handling for push failures
  - Mock docker client push method

- [ ] T021 [P] [US2] Integration test for image pushing in tests/integration/deploy/test_push_image.py
  - Spec details: @spec.md L:37-40 (acceptance scenarios)
  - Test push to local registry (use registry:2 container)
  - Test --dry-run flag behavior
  - Test error when no image exists locally
  - Requires Docker daemon (mark with @pytest.mark.docker)

### Implementation for User Story 2

- [ ] T022 [US2] Implement RegistryPusher class in src/holodeck/deploy/pusher.py
  - Research details: @research.md L:70-88 (push patterns)
  - Implement push() method with streaming progress
  - Support authentication from environment variables
  - Support credentials from ~/.docker/config.json
  - Handle push errors with clear messages

- [ ] T023 [US2] Implement `holodeck deploy push` CLI command in src/holodeck/cli/commands/deploy.py
  - Spec details: @spec.md L:113 (FR-016)
  - Implement push subcommand
  - Validate image exists locally before push
  - Add --dry-run support
  - Output pushed image URI on success

- [ ] T024 [US2] Add registry credential resolution
  - Research details: @research.md L:132-147 (authentication patterns)
  - Support HOLODECK_REGISTRY_USERNAME, HOLODECK_REGISTRY_PASSWORD
  - Support credentials_env_prefix from config
  - Fall back to docker config.json

- [ ] T025 [US2] Add validation for missing local image
  - Spec details: @spec.md L:40 (acceptance scenario 4)
  - Clear error message instructing to run build first
  - Exit code 2 (configuration error)

**Checkpoint**: `holodeck deploy push` works with built images - can push to registries

---

## Phase 5: User Story 3 - Deploy to Cloud Platform (Priority: P3)

**Goal**: Enable developers to deploy containerized agents to cloud platforms using `holodeck deploy run`

**Independent Test**: Run `holodeck deploy run agent.yaml` and verify the agent is accessible at the deployment URL.

**Spec Reference**: @spec.md L:44-59 (User Story 3 acceptance scenarios)

### Tests for User Story 3

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T026 [P] [US3] Unit tests for BaseDeployer interface in tests/unit/deploy/test_deployers_base.py
  - Plan details: @plan.md L:220-253 (deployer interface)
  - Test abstract method signatures
  - Test common validation logic

- [ ] T027 [P] [US3] Unit tests for AWS App Runner deployer in tests/unit/deploy/test_deployers_aws.py
  - Research details: @research.md L:156-231 (AWS patterns)
  - Mock boto3 client
  - Test create_service() call structure
  - Test get_status() parsing
  - Test destroy() method
  - Add mock responses in tests/fixtures/deploy/mock_responses/aws/

- [ ] T028 [P] [US3] Unit tests for GCP Cloud Run deployer in tests/unit/deploy/test_deployers_gcp.py
  - Research details: @research.md L:244-330 (GCP patterns)
  - Mock google-cloud-run client
  - Test create_service() with LRO handling
  - Test get_status() and destroy()
  - Add mock responses in tests/fixtures/deploy/mock_responses/gcp/

- [ ] T029 [P] [US3] Unit tests for Azure Container Apps deployer in tests/unit/deploy/test_deployers_azure.py
  - Research details: @research.md L:338-437 (Azure patterns)
  - Mock azure-mgmt-appcontainers client
  - Test begin_create_or_update() with poller
  - Test get_status() and destroy()
  - Add mock responses in tests/fixtures/deploy/mock_responses/azure/

### Implementation for User Story 3

- [ ] T030 [US3] Implement BaseDeployer abstract class in src/holodeck/deploy/deployers/base.py
  - Plan details: @plan.md L:220-253 (interface definition)
  - Define abstract methods: deploy(), get_status(), destroy(), stream_logs()
  - Add common validation and error handling

- [ ] T031 [US3] Implement AWSAppRunnerDeployer in src/holodeck/deploy/deployers/aws_apprunner.py
  - Research details: @research.md L:156-231 (implementation pattern)
  - Implement deploy() with boto3 create_service()
  - Implement get_status() with describe_service()
  - Implement destroy() with delete_service()
  - Handle missing boto3 with ImportError

- [ ] T032 [US3] Implement GCPCloudRunDeployer in src/holodeck/deploy/deployers/gcp_cloudrun.py
  - Research details: @research.md L:244-330 (implementation pattern)
  - Implement deploy() with ServicesClient.create_service()
  - Handle long-running operation (LRO) with operation.result()
  - Implement get_status() and destroy()
  - Handle missing google-cloud-run with ImportError

- [ ] T033 [US3] Implement AzureContainerAppsDeployer in src/holodeck/deploy/deployers/azure_containerapps.py
  - Research details: @research.md L:338-437 (implementation pattern)
  - Implement deploy() with begin_create_or_update()
  - Handle poller with result()
  - Implement get_status() and destroy()
  - Handle missing azure SDK with ImportError

- [ ] T034 [US3] Create deployer factory in src/holodeck/deploy/deployers/__init__.py
  - Return appropriate deployer based on provider config
  - Raise DeploymentError if SDK not installed
  - Provide clear installation instructions in error

- [ ] T035 [US3] Implement `holodeck deploy run` CLI command in src/holodeck/cli/commands/deploy.py
  - Spec details: @spec.md L:114 (FR-017)
  - Implement run subcommand
  - Validate image exists in registry
  - Inject environment variables from config
  - Output deployment URL and health check URL
  - Add --dry-run support

- [ ] T036 [US3] Implement `holodeck deploy status` CLI command in src/holodeck/cli/commands/deploy.py
  - Spec details: @spec.md L:117 (FR-029)
  - Query deployment status from cloud provider
  - Display URL, status, last updated timestamp

- [ ] T037 [US3] Implement `holodeck deploy destroy` CLI command in src/holodeck/cli/commands/deploy.py
  - Spec details: @spec.md L:118 (FR-030)
  - Confirm destruction (unless --force flag)
  - Call deployer.destroy()
  - Update local state file

- [ ] T038 [US3] Implement deployment state tracking in .holodeck/deployments.json
  - Data model details: @data-model.md L:353-373 (state file format)
  - Create/update state file on successful deploy
  - Use state file for status and destroy commands
  - Store service_id, url, status, timestamps, config_hash

**Checkpoint**: `holodeck deploy run/status/destroy` work - full cloud deployment lifecycle

---

## Phase 6: User Story 4 - Full Pipeline Deployment (Priority: P4)

**Goal**: Enable developers to run build, push, and deploy in a single command using `holodeck deploy`

**Independent Test**: Run `holodeck deploy agent.yaml` and verify the agent is accessible at the deployment URL.

**Spec Reference**: @spec.md L:62-75 (User Story 4 acceptance scenarios)

### Tests for User Story 4

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T039 [P] [US4] Integration test for full pipeline in tests/integration/deploy/test_full_pipeline.py
  - Spec details: @spec.md L:71-74 (acceptance scenarios)
  - Test sequence: build → push → run
  - Test --dry-run shows all steps
  - Test failure in build skips push and run
  - Mock cloud deployment for CI

### Implementation for User Story 4

- [ ] T040 [US4] Implement full pipeline in deploy group's default behavior in src/holodeck/cli/commands/deploy.py
  - Plan details: @plan.md L:141-151 (invoke_without_command pattern)
  - When no subcommand: invoke build, push, run in sequence
  - Stop on first failure with clear error
  - Show progress for each step
  - Add --dry-run support for full pipeline

- [ ] T041 [US4] Add progress indicators for pipeline steps
  - Spec details: @spec.md L:128 (FR-024)
  - Use spinner/progress indicator pattern from test runner
  - Show current step (Building → Pushing → Deploying)
  - Support --quiet for CI/CD (no progress, only errors)

**Checkpoint**: `holodeck deploy` runs complete pipeline - single command to production

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Quality improvements, documentation, and CI/CD integration

- [ ] T042 [P] Add OpenTelemetry spans for deploy operations
  - Spec details: @plan.md L:55-57 (observability)
  - Add spans: holodeck.deploy.build, holodeck.deploy.push, holodeck.deploy.run
  - Include relevant attributes (image_name, registry, provider)

- [ ] T043 [P] Add comprehensive logging throughout deploy module
  - Use holodeck logging conventions
  - Log at appropriate levels (DEBUG for progress, INFO for results, ERROR for failures)

- [ ] T044 [P] Create integration test with real Docker in tests/integration/deploy/test_e2e_docker.py
  - Full build+push test with local registry
  - Mark with @pytest.mark.docker and @pytest.mark.slow
  - Skip in CI if Docker not available

- [x] T045 Update holodeck --help and docs with deploy command
  - Add deploy command to CLI help
  - Update README.md if needed
  - Created comprehensive deployment guide at docs/guides/deployment.md
  - Added navigation entry to mkdocs.yml

- [ ] T046 Run and validate quickstart.md scenarios
  - Quickstart details: @quickstart.md L:1-396
  - Verify all commands work as documented
  - Update any incorrect examples

- [ ] T047 [P] Add type hints and run mypy on deploy module
  - Ensure all public functions have type annotations
  - Fix any mypy errors

- [ ] T048 [P] Run security scan (bandit) on deploy module
  - Check for hardcoded credentials
  - Verify no secrets logged

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **Base Image (Phase 2b)**: Depends on Phase 2 - BLOCKS full testing of Phase 3
- **User Story 1 (Phase 3)**: Depends on Foundational - can start after Phase 2, full testing needs Phase 2b
- **User Story 2 (Phase 4)**: Depends on Foundational - can start after Phase 2, integrates with US1
- **User Story 3 (Phase 5)**: Depends on Foundational - can start after Phase 2, integrates with US1+US2
- **User Story 4 (Phase 6)**: Depends on US1, US2, US3 - orchestrates all three
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

```
Phase 1: Setup
    ↓
Phase 2: Foundational (Pydantic models, Dockerfile generation)
    ↓
Phase 2b: Base Image (build & publish holodeck-base to ghcr.io)
    ↓
    ├── Phase 3: US1 - Build (can start after Phase 2, full testing needs Phase 2b)
    ↓
    ├── Phase 4: US2 - Push (can start after Phase 2, needs built image from US1 to test fully)
    ↓
    ├── Phase 5: US3 - Deploy (can start after Phase 2, needs pushed image from US2 to test fully)
    ↓
Phase 6: US4 - Full Pipeline (requires US1 + US2 + US3)
    ↓
Phase 7: Polish
```

### Within Each User Story (TDD Pattern)

1. Tests FIRST - must FAIL before implementation
2. Models before services
3. Services/utilities before CLI commands
4. Core implementation before integration
5. Verify tests PASS after implementation

### Parallel Opportunities

**Phase 1 (Setup)**:
- T002, T003, T004, T005 can run in parallel

**Phase 2 (Foundational)**:
- T006, T007 (tests) can run in parallel
- T008, T009, T010, T011 can be parallelized after tests written

**Phase 2b (Base Image)**:
- T012a, T012b can run in parallel (workflow and build script)
- T012c, T012d are sequential (test then publish)

**Phase 3-5 (User Stories)**:
- Tests within each story can run in parallel (T013-T015, T020-T021, T026-T029)
- US1, US2, US3 core implementation can start in parallel after Phase 2
- Full integration testing requires sequential completion

**Phase 7 (Polish)**:
- T042, T043, T044, T047, T048 can run in parallel

---

## Parallel Example: Phase 2 Foundational

```bash
# Launch foundational tests in parallel:
Task: T006 "Unit tests for Pydantic deployment models"
Task: T007 "Unit tests for Dockerfile generation"

# Then implement in parallel (after tests fail):
Task: T008 "Implement Pydantic models"
Task: T009 "Implement Dockerfile generator"
Task: T010 "Create HoloDeck base image Dockerfile"
Task: T011 "Create container entrypoint script"
```

---

## Parallel Example: User Story 3 Cloud Deployers

```bash
# Launch all deployer tests in parallel:
Task: T026 "Unit tests for BaseDeployer interface"
Task: T027 "Unit tests for AWS App Runner deployer"
Task: T028 "Unit tests for GCP Cloud Run deployer"
Task: T029 "Unit tests for Azure Container Apps deployer"

# Then implement deployers in parallel:
Task: T030 "Implement BaseDeployer abstract class"
Task: T031 "Implement AWSAppRunnerDeployer"
Task: T032 "Implement GCPCloudRunDeployer"
Task: T033 "Implement AzureContainerAppsDeployer"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T012)
3. Complete Phase 2b: Base Image (T012a-T012d) - build & publish holodeck-base
4. Complete Phase 3: User Story 1 - Build (T013-T019)
5. **STOP and VALIDATE**: `holodeck deploy build` works independently
6. Demo: Developer can build container images from agent.yaml

### Incremental Delivery

1. Setup + Foundational → Core infrastructure ready
2. Base Image (Phase 2b) → Publish holodeck-base to ghcr.io
3. Add User Story 1 (Build) → Test independently → Demo (developers can build images!)
4. Add User Story 2 (Push) → Test independently → Demo (developers can push to registries!)
5. Add User Story 3 (Deploy) → Test independently → Demo (developers can deploy to cloud!)
6. Add User Story 4 (Pipeline) → Test full workflow → Demo (single command deployment!)
7. Polish phase → Production-ready quality

### Parallel Team Strategy

With multiple developers:

1. All: Complete Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Build)
   - Developer B: User Story 2 (Push) - mock build output initially
   - Developer C: User Story 3 (Deploy) - mock push output initially
3. Integration: Combine and test full pipeline
4. Developer D: User Story 4 (Pipeline) once US1-3 are ready

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- TDD approach: Write tests first, verify they FAIL, then implement
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Cloud deployers use optional dependencies - handle ImportError gracefully
