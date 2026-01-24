#!/bin/bash
# Build script for HoloDeck base image
#
# Usage:
#   ./scripts/build-base-image.sh                       # Build locally
#   ./scripts/build-base-image.sh --tag 0.1.0           # Build with version tag
#   ./scripts/build-base-image.sh --push --tag 0.1.0    # Build and push
#   ./scripts/build-base-image.sh --platform linux/arm64  # Build for specific platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="ghcr.io/justinbarias/holodeck-base"
TAG="latest"
PUSH=false
PLATFORM=""
HOLODECK_VERSION=""
NO_CACHE=false

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
usage() {
    cat << EOF
HoloDeck Base Image Build Script

Usage: $(basename "$0") [OPTIONS]

Options:
    --tag TAG           Image tag (default: latest)
    --push              Push image to registry after build
    --platform PLATFORM Target platform (e.g., linux/amd64, linux/arm64)
    --version VERSION   HoloDeck version to install (default: latest from PyPI)
    --no-cache          Build without cache
    -h, --help          Show this help message

Examples:
    $(basename "$0")                           # Build locally with tag 'latest'
    $(basename "$0") --tag 0.1.0               # Build with specific tag
    $(basename "$0") --push --tag 0.1.0        # Build and push to registry
    $(basename "$0") --platform linux/arm64    # Build for ARM64
    $(basename "$0") --version 0.1.0           # Install specific holodeck version
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --version)
            HOLODECK_VERSION="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_PATH="${PROJECT_ROOT}/docker/Dockerfile"
CONTEXT_PATH="${PROJECT_ROOT}/docker"

# Verify Dockerfile exists
if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    print_error "Dockerfile not found at ${DOCKERFILE_PATH}"
    exit 1
fi

# Build Docker command
BUILD_CMD="docker build"

# Add platform if specified
if [[ -n "${PLATFORM}" ]]; then
    BUILD_CMD+=" --platform ${PLATFORM}"
fi

# Add no-cache if specified
if [[ "${NO_CACHE}" == true ]]; then
    BUILD_CMD+=" --no-cache"
fi

# Add build args (only add version if it's a specific version, not "latest")
if [[ -n "${HOLODECK_VERSION}" && "${HOLODECK_VERSION}" != "latest" ]]; then
    BUILD_CMD+=" --build-arg HOLODECK_VERSION=${HOLODECK_VERSION}"
fi

# Add tags
BUILD_CMD+=" -t ${IMAGE_NAME}:${TAG}"

# Add Dockerfile and context
BUILD_CMD+=" -f ${DOCKERFILE_PATH} ${CONTEXT_PATH}"

# Print build info
print_info "Building HoloDeck base image"
print_info "  Image: ${IMAGE_NAME}:${TAG}"
print_info "  Dockerfile: ${DOCKERFILE_PATH}"
if [[ -n "${PLATFORM}" ]]; then
    print_info "  Platform: ${PLATFORM}"
fi
if [[ -n "${HOLODECK_VERSION}" ]]; then
    print_info "  HoloDeck Version: ${HOLODECK_VERSION}"
fi
echo ""

# Run build
print_info "Running: ${BUILD_CMD}"
echo ""

if eval "${BUILD_CMD}"; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

# Push if requested
if [[ "${PUSH}" == true ]]; then
    echo ""
    print_info "Pushing image to registry..."

    # Check if logged in to ghcr.io
    if ! docker info 2>/dev/null | grep -q "ghcr.io"; then
        print_warning "You may need to login to ghcr.io first:"
        print_warning "  echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
    fi

    if docker push "${IMAGE_NAME}:${TAG}"; then
        print_success "Image pushed successfully: ${IMAGE_NAME}:${TAG}"
    else
        print_error "Push failed"
        exit 1
    fi
fi

echo ""
print_success "Done!"
print_info "To run the image:"
print_info "  docker run -v \$(pwd)/agent.yaml:/app/agent.yaml -p 8080:8080 ${IMAGE_NAME}:${TAG}"
