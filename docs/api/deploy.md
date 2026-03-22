# Deployment Subsystem

The deployment subsystem packages HoloDeck agents as container images and deploys
them to cloud providers. It covers three concerns: Dockerfile generation, Docker
image building, and cloud deployment with state tracking.

!!! note "Optional dependency"
    `ContainerBuilder` and `BuildResult` require the **docker** Python package.
    Install it with `pip install holodeck-ai[deploy]`. The symbols are lazily
    imported so the rest of the library works without Docker installed.

---

## Package entry point

::: holodeck.deploy
    options:
      docstring_style: google
      show_source: true
      members:
        - generate_dockerfile
        - generate_tag
        - get_oci_labels

---

## `holodeck.deploy.dockerfile` -- Dockerfile generation

Generates Dockerfiles from Jinja2 templates, embedding OCI labels, environment
variables, instruction files, and optional Node.js installation for Claude agents.

::: holodeck.deploy.dockerfile.generate_dockerfile
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.deploy.builder` -- Container image building

Builds Docker images via the Docker SDK and provides helpers for tag generation
and OCI label creation.

### BuildResult

::: holodeck.deploy.builder.BuildResult
    options:
      docstring_style: google
      show_source: true
      members:
        - from_image

### ContainerBuilder

::: holodeck.deploy.builder.ContainerBuilder
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - build

### generate_tag

::: holodeck.deploy.builder.generate_tag
    options:
      docstring_style: google
      show_source: true

### get_oci_labels

::: holodeck.deploy.builder.get_oci_labels
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.deploy.state` -- Deployment state tracking

Persists deployment records to a JSON file under `.holodeck/deployments.json`
next to the agent configuration. Tracks creation and update timestamps and
computes deterministic config hashes for drift detection.

### get_state_path

::: holodeck.deploy.state.get_state_path
    options:
      docstring_style: google
      show_source: true

### compute_config_hash

::: holodeck.deploy.state.compute_config_hash
    options:
      docstring_style: google
      show_source: true

### load_state

::: holodeck.deploy.state.load_state
    options:
      docstring_style: google
      show_source: true

### save_state

::: holodeck.deploy.state.save_state
    options:
      docstring_style: google
      show_source: true

### get_deployment_record

::: holodeck.deploy.state.get_deployment_record
    options:
      docstring_style: google
      show_source: true

### update_deployment_record

::: holodeck.deploy.state.update_deployment_record
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.deploy.deployers` -- Cloud deployers

Factory module that instantiates the correct deployer based on cloud provider.

### create_deployer

::: holodeck.deploy.deployers.create_deployer
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.deploy.deployers.base` -- Base deployer interface

### BaseDeployer

::: holodeck.deploy.deployers.base.BaseDeployer
    options:
      docstring_style: google
      show_source: true
      members:
        - deploy
        - get_status
        - destroy
        - stream_logs

---

## `holodeck.deploy.deployers.azure_containerapps` -- Azure Container Apps

### AzureContainerAppsDeployer

::: holodeck.deploy.deployers.azure_containerapps.AzureContainerAppsDeployer
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - deploy
        - get_status
        - destroy
        - stream_logs
