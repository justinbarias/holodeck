#!/bin/bash
# HoloDeck Container Entrypoint
#
# This script serves as the entrypoint for HoloDeck agent containers.
# It validates configuration and starts the HoloDeck serve command.
#
# Environment Variables:
#   HOLODECK_PORT - Port to serve on (default: 8080)
#   HOLODECK_PROTOCOL - Protocol type: rest, ag-ui, both (default: rest)
#   HOLODECK_AGENT_CONFIG - Path to agent.yaml (default: /app/agent.yaml)
#   HOLODECK_LOG_LEVEL - Log level: debug, info, warning, error (default: info)

set -e

# Default configuration
HOLODECK_PORT="${HOLODECK_PORT:-8080}"
HOLODECK_PROTOCOL="${HOLODECK_PROTOCOL:-rest}"
HOLODECK_AGENT_CONFIG="${HOLODECK_AGENT_CONFIG:-/app/agent.yaml}"
HOLODECK_LOG_LEVEL="${HOLODECK_LOG_LEVEL:-info}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate agent configuration file exists
validate_config() {
    if [ ! -f "${HOLODECK_AGENT_CONFIG}" ]; then
        log_error "Agent configuration file not found: ${HOLODECK_AGENT_CONFIG}"
        log_error "Please ensure agent.yaml is mounted or copied to the container."
        exit 1
    fi
    log_info "Found agent configuration: ${HOLODECK_AGENT_CONFIG}"
}

# Validate protocol value
validate_protocol() {
    case "${HOLODECK_PROTOCOL}" in
        rest|ag-ui|both)
            log_info "Protocol: ${HOLODECK_PROTOCOL}"
            ;;
        *)
            log_error "Invalid protocol: ${HOLODECK_PROTOCOL}"
            log_error "Must be one of: rest, ag-ui, both"
            exit 1
            ;;
    esac
}

# Validate port is a number
validate_port() {
    if ! [[ "${HOLODECK_PORT}" =~ ^[0-9]+$ ]]; then
        log_error "Invalid port: ${HOLODECK_PORT}"
        log_error "Port must be a number"
        exit 1
    fi

    if [ "${HOLODECK_PORT}" -lt 1 ] || [ "${HOLODECK_PORT}" -gt 65535 ]; then
        log_error "Invalid port range: ${HOLODECK_PORT}"
        log_error "Port must be between 1 and 65535"
        exit 1
    fi

    log_info "Port: ${HOLODECK_PORT}"
}

# Print startup banner
print_banner() {
    echo ""
    echo "  _   _       _       ____            _    "
    echo " | | | | ___ | | ___ |  _ \  ___  ___| | __"
    echo " | |_| |/ _ \| |/ _ \| | | |/ _ \/ __| |/ /"
    echo " |  _  | (_) | | (_) | |_| |  __/ (__|   < "
    echo " |_| |_|\___/|_|\___/|____/ \___|\___|_|\_\\"
    echo ""
    echo " HoloDeck AI Agent Container"
    echo " =============================="
    echo ""
}

# Main execution
main() {
    print_banner

    log_info "Starting HoloDeck agent..."
    log_info "Log level: ${HOLODECK_LOG_LEVEL}"

    # Validate configuration
    validate_config
    validate_protocol
    validate_port

    echo ""
    log_info "Launching HoloDeck serve..."
    echo ""

    # Build command arguments
    CMD_ARGS="--config ${HOLODECK_AGENT_CONFIG}"
    CMD_ARGS="${CMD_ARGS} --port ${HOLODECK_PORT}"
    CMD_ARGS="${CMD_ARGS} --protocol ${HOLODECK_PROTOCOL}"

    # Execute holodeck serve command
    # Using exec to replace shell process with holodeck
    exec holodeck serve ${CMD_ARGS}
}

# Run main function
main "$@"
