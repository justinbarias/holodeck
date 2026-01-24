"""Unit tests for deployment Pydantic models.

TDD tests - these should fail until T008 implements the models.
"""

import pytest
from pydantic import ValidationError


class TestTagStrategy:
    """Tests for TagStrategy enum."""

    def test_valid_tag_strategies(self) -> None:
        """Test all valid tag strategy values."""
        from holodeck.models.deployment import TagStrategy

        assert TagStrategy.GIT_SHA.value == "git_sha"
        assert TagStrategy.GIT_TAG.value == "git_tag"
        assert TagStrategy.LATEST.value == "latest"
        assert TagStrategy.CUSTOM.value == "custom"


class TestCloudProvider:
    """Tests for CloudProvider enum."""

    def test_valid_cloud_providers(self) -> None:
        """Test all valid cloud provider values."""
        from holodeck.models.deployment import CloudProvider

        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.AZURE.value == "azure"


class TestRuntimeType:
    """Tests for RuntimeType enum."""

    def test_valid_runtime_types(self) -> None:
        """Test all valid runtime type values."""
        from holodeck.models.deployment import RuntimeType

        assert RuntimeType.CONTAINER.value == "container"


class TestProtocolType:
    """Tests for ProtocolType enum."""

    def test_valid_protocol_types(self) -> None:
        """Test all valid protocol type values."""
        from holodeck.models.deployment import ProtocolType

        assert ProtocolType.REST.value == "rest"
        assert ProtocolType.AG_UI.value == "ag-ui"
        assert ProtocolType.BOTH.value == "both"


class TestRegistryConfig:
    """Tests for RegistryConfig model."""

    def test_minimal_registry_config(self) -> None:
        """Test minimal valid registry configuration."""
        from holodeck.models.deployment import RegistryConfig

        config = RegistryConfig(
            url="ghcr.io",
            repository="my-org/my-agent",
        )
        assert config.url == "ghcr.io"
        assert config.repository == "my-org/my-agent"
        assert config.tag_strategy.value == "git_sha"  # default
        assert config.custom_tag is None

    def test_registry_config_with_custom_tag(self) -> None:
        """Test registry config with custom tag strategy."""
        from holodeck.models.deployment import RegistryConfig, TagStrategy

        config = RegistryConfig(
            url="docker.io",
            repository="myrepo/myagent",
            tag_strategy=TagStrategy.CUSTOM,
            custom_tag="v1.0.0",
        )
        assert config.tag_strategy == TagStrategy.CUSTOM
        assert config.custom_tag == "v1.0.0"

    def test_registry_config_custom_tag_required_when_custom_strategy(self) -> None:
        """Test that custom_tag is required when tag_strategy is CUSTOM."""
        from holodeck.models.deployment import RegistryConfig, TagStrategy

        with pytest.raises(ValidationError) as exc_info:
            RegistryConfig(
                url="ghcr.io",
                repository="org/agent",
                tag_strategy=TagStrategy.CUSTOM,
                # custom_tag is missing
            )
        assert "custom_tag" in str(exc_info.value)

    def test_registry_config_repository_pattern(self) -> None:
        """Test repository name pattern validation."""
        from holodeck.models.deployment import RegistryConfig

        # Valid patterns
        RegistryConfig(url="ghcr.io", repository="org/agent")
        RegistryConfig(url="ghcr.io", repository="my-org/my-agent")
        RegistryConfig(url="ghcr.io", repository="org123/agent_name")

        # Invalid patterns
        with pytest.raises(ValidationError):
            RegistryConfig(url="ghcr.io", repository="invalid repo")  # spaces

    def test_registry_config_credentials_env_prefix(self) -> None:
        """Test credentials_env_prefix configuration."""
        from holodeck.models.deployment import RegistryConfig

        config = RegistryConfig(
            url="private.registry.io",
            repository="org/agent",
            credentials_env_prefix="PRIVATE_REGISTRY",
        )
        assert config.credentials_env_prefix == "PRIVATE_REGISTRY"

    def test_registry_config_forbids_extra_fields(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.deployment import RegistryConfig

        with pytest.raises(ValidationError):
            RegistryConfig(
                url="ghcr.io",
                repository="org/agent",
                unknown_field="value",  # type: ignore[call-arg]
            )


class TestAWSAppRunnerConfig:
    """Tests for AWSAppRunnerConfig model."""

    def test_minimal_aws_config(self) -> None:
        """Test minimal valid AWS App Runner configuration."""
        from holodeck.models.deployment import AWSAppRunnerConfig

        config = AWSAppRunnerConfig(region="us-east-1")
        assert config.region == "us-east-1"
        assert config.cpu == 1  # default
        assert config.memory == 2048  # default

    def test_aws_config_with_all_options(self) -> None:
        """Test AWS config with all options specified."""
        from holodeck.models.deployment import AWSAppRunnerConfig

        config = AWSAppRunnerConfig(
            region="us-west-2",
            cpu=2,
            memory=4096,
            ecr_role_arn="arn:aws:iam::123456789012:role/AppRunnerECRAccess",
            health_check_path="/health",
            auto_scaling_min=1,
            auto_scaling_max=10,
        )
        assert config.region == "us-west-2"
        assert config.cpu == 2
        assert config.memory == 4096
        assert (
            config.ecr_role_arn == "arn:aws:iam::123456789012:role/AppRunnerECRAccess"
        )
        assert config.health_check_path == "/health"
        assert config.auto_scaling_min == 1
        assert config.auto_scaling_max == 10

    def test_aws_config_ecr_role_arn_pattern(self) -> None:
        """Test ECR role ARN pattern validation."""
        from holodeck.models.deployment import AWSAppRunnerConfig

        # Valid ARN
        AWSAppRunnerConfig(
            region="us-east-1",
            ecr_role_arn="arn:aws:iam::123456789012:role/MyRole",
        )

        # Invalid ARN
        with pytest.raises(ValidationError) as exc_info:
            AWSAppRunnerConfig(
                region="us-east-1",
                ecr_role_arn="not-a-valid-arn",
            )
        assert "ecr_role_arn" in str(exc_info.value)

    def test_aws_config_cpu_memory_combinations(self) -> None:
        """Test valid CPU and memory combinations."""
        from holodeck.models.deployment import AWSAppRunnerConfig

        # Valid combinations
        valid_combos = [
            (1, 2048),
            (1, 3072),
            (1, 4096),
            (2, 4096),
            (4, 8192),
            (4, 12288),
        ]
        for cpu, memory in valid_combos:
            config = AWSAppRunnerConfig(region="us-east-1", cpu=cpu, memory=memory)
            assert config.cpu == cpu
            assert config.memory == memory

    def test_aws_config_scaling_range_validation(self) -> None:
        """Test auto scaling range validation."""
        from holodeck.models.deployment import AWSAppRunnerConfig

        # Min must be <= Max
        with pytest.raises(ValidationError) as exc_info:
            AWSAppRunnerConfig(
                region="us-east-1",
                auto_scaling_min=10,
                auto_scaling_max=5,
            )
        assert "auto_scaling" in str(exc_info.value).lower()


class TestGCPCloudRunConfig:
    """Tests for GCPCloudRunConfig model."""

    def test_minimal_gcp_config(self) -> None:
        """Test minimal valid GCP Cloud Run configuration."""
        from holodeck.models.deployment import GCPCloudRunConfig

        config = GCPCloudRunConfig(project_id="my-project")
        assert config.project_id == "my-project"
        assert config.region == "us-central1"  # default
        assert config.memory == "512Mi"  # default
        assert config.cpu == 1  # default

    def test_gcp_config_with_all_options(self) -> None:
        """Test GCP config with all options specified."""
        from holodeck.models.deployment import GCPCloudRunConfig

        config = GCPCloudRunConfig(
            project_id="my-project-123",
            region="europe-west1",
            memory="2Gi",
            cpu=2,
            concurrency=100,
            timeout=300,
            min_instances=1,
            max_instances=10,
        )
        assert config.project_id == "my-project-123"
        assert config.region == "europe-west1"
        assert config.memory == "2Gi"
        assert config.cpu == 2
        assert config.concurrency == 100
        assert config.timeout == 300
        assert config.min_instances == 1
        assert config.max_instances == 10

    def test_gcp_config_project_id_pattern(self) -> None:
        """Test project ID pattern validation."""
        from holodeck.models.deployment import GCPCloudRunConfig

        # Valid project IDs
        GCPCloudRunConfig(project_id="my-project")
        GCPCloudRunConfig(project_id="my-project-123")
        GCPCloudRunConfig(project_id="project123")

        # Invalid project IDs
        with pytest.raises(ValidationError):
            GCPCloudRunConfig(project_id="My_Project")  # underscores, uppercase

    def test_gcp_config_memory_pattern(self) -> None:
        """Test memory pattern validation (e.g., 512Mi, 2Gi)."""
        from holodeck.models.deployment import GCPCloudRunConfig

        # Valid memory patterns
        GCPCloudRunConfig(project_id="test-project", memory="512Mi")
        GCPCloudRunConfig(project_id="test-project", memory="1Gi")
        GCPCloudRunConfig(project_id="test-project", memory="4Gi")

        # Invalid memory patterns
        with pytest.raises(ValidationError):
            GCPCloudRunConfig(project_id="test-project", memory="2048")  # no unit
        with pytest.raises(ValidationError):
            GCPCloudRunConfig(project_id="test-project", memory="2GB")  # wrong unit

    def test_gcp_config_concurrency_range(self) -> None:
        """Test concurrency range validation (1-1000)."""
        from holodeck.models.deployment import GCPCloudRunConfig

        # Valid range
        GCPCloudRunConfig(project_id="test-project", concurrency=1)
        GCPCloudRunConfig(project_id="test-project", concurrency=1000)

        # Invalid range
        with pytest.raises(ValidationError):
            GCPCloudRunConfig(project_id="test-project", concurrency=0)
        with pytest.raises(ValidationError):
            GCPCloudRunConfig(project_id="test-project", concurrency=1001)


class TestAzureContainerAppsConfig:
    """Tests for AzureContainerAppsConfig model."""

    def test_minimal_azure_config(self) -> None:
        """Test minimal valid Azure Container Apps configuration."""
        from holodeck.models.deployment import AzureContainerAppsConfig

        config = AzureContainerAppsConfig(
            subscription_id="00000000-0000-0000-0000-000000000000",
            resource_group="my-rg",
        )
        assert config.subscription_id == "00000000-0000-0000-0000-000000000000"
        assert config.resource_group == "my-rg"
        assert config.location == "eastus"  # default

    def test_azure_config_with_all_options(self) -> None:
        """Test Azure config with all options specified."""
        from holodeck.models.deployment import AzureContainerAppsConfig

        config = AzureContainerAppsConfig(
            subscription_id="11111111-1111-1111-1111-111111111111",
            resource_group="my-rg",
            environment_name="my-env",
            location="westeurope",
            cpu=1.0,
            memory="2Gi",
            ingress_external=True,
            min_replicas=1,
            max_replicas=5,
        )
        assert config.subscription_id == "11111111-1111-1111-1111-111111111111"
        assert config.resource_group == "my-rg"
        assert config.environment_name == "my-env"
        assert config.location == "westeurope"
        assert config.cpu == 1.0
        assert config.memory == "2Gi"
        assert config.ingress_external is True
        assert config.min_replicas == 1
        assert config.max_replicas == 5

    def test_azure_config_subscription_id_uuid_format(self) -> None:
        """Test subscription ID UUID format validation."""
        from holodeck.models.deployment import AzureContainerAppsConfig

        # Valid UUID
        AzureContainerAppsConfig(
            subscription_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            resource_group="rg",
        )

        # Invalid UUID
        with pytest.raises(ValidationError) as exc_info:
            AzureContainerAppsConfig(
                subscription_id="not-a-uuid",
                resource_group="rg",
            )
        assert "subscription_id" in str(exc_info.value)

    def test_azure_config_cpu_values(self) -> None:
        """Test valid CPU values (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)."""
        from holodeck.models.deployment import AzureContainerAppsConfig

        valid_cpus = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        for cpu in valid_cpus:
            config = AzureContainerAppsConfig(
                subscription_id="00000000-0000-0000-0000-000000000000",
                resource_group="rg",
                cpu=cpu,
            )
            assert config.cpu == cpu


class TestCloudTargetConfig:
    """Tests for CloudTargetConfig discriminated union."""

    def test_aws_target_config(self) -> None:
        """Test AWS target configuration."""
        from holodeck.models.deployment import CloudProvider, CloudTargetConfig

        config = CloudTargetConfig(
            provider=CloudProvider.AWS,
            aws={"region": "us-east-1", "cpu": 1, "memory": 2048},
        )
        assert config.provider == CloudProvider.AWS
        assert config.aws is not None
        assert config.aws.region == "us-east-1"

    def test_gcp_target_config(self) -> None:
        """Test GCP target configuration."""
        from holodeck.models.deployment import CloudProvider, CloudTargetConfig

        config = CloudTargetConfig(
            provider=CloudProvider.GCP,
            gcp={"project_id": "my-project"},
        )
        assert config.provider == CloudProvider.GCP
        assert config.gcp is not None
        assert config.gcp.project_id == "my-project"

    def test_azure_target_config(self) -> None:
        """Test Azure target configuration."""
        from holodeck.models.deployment import CloudProvider, CloudTargetConfig

        config = CloudTargetConfig(
            provider=CloudProvider.AZURE,
            azure={
                "subscription_id": "00000000-0000-0000-0000-000000000000",
                "resource_group": "my-rg",
            },
        )
        assert config.provider == CloudProvider.AZURE
        assert config.azure is not None
        assert config.azure.resource_group == "my-rg"

    def test_provider_config_mismatch_validation(self) -> None:
        """Test that provider must match the provided config."""
        from holodeck.models.deployment import CloudProvider, CloudTargetConfig

        with pytest.raises(ValidationError) as exc_info:
            CloudTargetConfig(
                provider=CloudProvider.AWS,
                gcp={"project_id": "my-project"},  # GCP config with AWS provider
            )
        assert (
            "aws" in str(exc_info.value).lower()
            or "provider" in str(exc_info.value).lower()
        )

    def test_missing_provider_config_validation(self) -> None:
        """Test that provider-specific config is required."""
        from holodeck.models.deployment import CloudProvider, CloudTargetConfig

        with pytest.raises(ValidationError):
            CloudTargetConfig(
                provider=CloudProvider.AWS,
                # aws config is missing
            )


class TestDeploymentConfig:
    """Tests for DeploymentConfig model."""

    def test_minimal_deployment_config(self) -> None:
        """Test minimal valid deployment configuration."""
        from holodeck.models.deployment import DeploymentConfig

        config = DeploymentConfig(
            registry={"url": "ghcr.io", "repository": "org/agent"},
            target={
                "provider": "aws",
                "aws": {"region": "us-east-1"},
            },
        )
        assert config.registry.url == "ghcr.io"
        assert config.target.provider.value == "aws"
        assert config.port == 8080  # default
        assert config.protocol.value == "rest"  # default

    def test_deployment_config_with_all_options(self) -> None:
        """Test deployment config with all options."""
        from holodeck.models.deployment import (
            DeploymentConfig,
            ProtocolType,
            RuntimeType,
        )

        config = DeploymentConfig(
            runtime=RuntimeType.CONTAINER,
            registry={
                "url": "ghcr.io",
                "repository": "org/agent",
                "tag_strategy": "custom",
                "custom_tag": "v1.0.0",
            },
            target={
                "provider": "gcp",
                "gcp": {"project_id": "my-project", "memory": "1Gi"},
            },
            protocol=ProtocolType.BOTH,
            port=3000,
            environment={"DEBUG": "true", "LOG_LEVEL": "debug"},
        )
        assert config.runtime == RuntimeType.CONTAINER
        assert config.registry.custom_tag == "v1.0.0"
        assert config.target.gcp is not None
        assert config.target.gcp.memory == "1Gi"
        assert config.protocol == ProtocolType.BOTH
        assert config.port == 3000
        assert config.environment == {"DEBUG": "true", "LOG_LEVEL": "debug"}

    def test_deployment_config_port_range(self) -> None:
        """Test port range validation (1-65535)."""
        from holodeck.models.deployment import DeploymentConfig

        # Valid ports
        DeploymentConfig(
            registry={"url": "ghcr.io", "repository": "org/agent"},
            target={"provider": "aws", "aws": {"region": "us-east-1"}},
            port=1,
        )
        DeploymentConfig(
            registry={"url": "ghcr.io", "repository": "org/agent"},
            target={"provider": "aws", "aws": {"region": "us-east-1"}},
            port=65535,
        )

        # Invalid ports
        with pytest.raises(ValidationError):
            DeploymentConfig(
                registry={"url": "ghcr.io", "repository": "org/agent"},
                target={"provider": "aws", "aws": {"region": "us-east-1"}},
                port=0,
            )
        with pytest.raises(ValidationError):
            DeploymentConfig(
                registry={"url": "ghcr.io", "repository": "org/agent"},
                target={"provider": "aws", "aws": {"region": "us-east-1"}},
                port=65536,
            )

    def test_deployment_config_environment_dict(self) -> None:
        """Test environment variables dictionary."""
        from holodeck.models.deployment import DeploymentConfig

        config = DeploymentConfig(
            registry={"url": "ghcr.io", "repository": "org/agent"},
            target={"provider": "aws", "aws": {"region": "us-east-1"}},
            environment={
                "API_KEY": "${API_KEY}",
                "DEBUG": "false",
                "MAX_RETRIES": "3",
            },
        )
        assert len(config.environment) == 3
        assert config.environment["API_KEY"] == "${API_KEY}"

    def test_deployment_config_forbids_extra_fields(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.deployment import DeploymentConfig

        with pytest.raises(ValidationError):
            DeploymentConfig(
                registry={"url": "ghcr.io", "repository": "org/agent"},
                target={"provider": "aws", "aws": {"region": "us-east-1"}},
                unknown_field="value",  # type: ignore[call-arg]
            )
