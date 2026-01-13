#!/usr/bin/env bash
# Unified management script for all Docker services
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${SCRIPT_DIR}/backups"
MONITOR_PID_FILE="${SCRIPT_DIR}/monitor.pid"
MONITOR_LOG_FILE="${SCRIPT_DIR}/monitor.log"
LOGS_DIR="${SCRIPT_DIR}/logs"
LOG_COLLECTOR_PID_FILE="${SCRIPT_DIR}/log_collector.pid"
RUNNING_MARKER="${SCRIPT_DIR}/.running"

# Load THALA_MODE from .env if not already set
if [[ -z "$THALA_MODE" && -f "${PROJECT_DIR}/.env" ]]; then
    THALA_MODE=$(grep -E "^THALA_MODE=" "${PROJECT_DIR}/.env" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'")
fi

# Use project venv Python if available, otherwise fall back to system python3
if [[ -f "${PROJECT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_DIR}/.venv/bin/python"
else
    PYTHON="python3"
fi

is_dev_mode() {
    [[ "$THALA_MODE" == "dev" ]]
}

# Service directories (order matters for startup - databases first)
SERVICES=(
    "elasticsearch-coherence"
    "elasticsearch-forgotten"
    "chroma"
    "zotero"
    "firecrawl"
)

# GPU services (require nvidia-container-toolkit)
GPU_SERVICES=(
    "marker"
)

# VPN services (require VPN credentials)
VPN_SERVICES=(
    "retrieve-academic"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; }

check_nvidia() {
    if ! command -v nvidia-smi &>/dev/null; then
        return 1
    fi
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        return 1
    fi
    return 0
}

check_vpn_config() {
    local svc="${1}"
    local env_file="${SCRIPT_DIR}/${svc}/.env"
    if [[ ! -f "$env_file" ]]; then
        return 1
    fi
    # Check for VPN credentials (supports both ExpressVPN and OpenVPN configs)
    if grep -qE "(EXPRESSVPN_USERNAME|OPENVPN_USER)" "$env_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

cmd_up() {
    log "Starting all services..."
    # Use --force-recreate to avoid stale Docker networks after unclean WSL shutdown
    for svc in "${SERVICES[@]}"; do
        log "Starting $svc..."
        docker compose -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" up -d --force-recreate
    done

    # Start GPU services if nvidia-container-toolkit is available
    if check_nvidia; then
        for svc in "${GPU_SERVICES[@]}"; do
            log "Starting GPU service: $svc..."
            docker compose -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" up -d --force-recreate
        done
    else
        warn "Skipping GPU services (nvidia-container-toolkit not available)"
        warn "GPU services: ${GPU_SERVICES[*]}"
    fi

    # Start VPN services if configured
    local vpn_started=0
    for svc in "${VPN_SERVICES[@]}"; do
        if [[ -d "${SCRIPT_DIR}/${svc}" ]] && check_vpn_config "$svc"; then
            log "Starting VPN service: $svc..."
            docker compose -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" up -d --force-recreate
            vpn_started=$((vpn_started + 1))
        fi
    done
    if [[ $vpn_started -eq 0 ]] && [[ ${#VPN_SERVICES[@]} -gt 0 ]]; then
        warn "Skipping VPN services (not configured or submodule not initialized)"
        warn "VPN services: ${VPN_SERVICES[*]}"
        warn "Run: git submodule update --init services/<service>"
    fi

    log "All services started."
    cmd_status

    # Start dev mode monitoring (metrics + logs) if THALA_MODE=dev
    log "Checking dev mode (THALA_MODE=$THALA_MODE)..."
    if is_dev_mode; then
        log "Dev mode detected - starting monitoring..."
        start_background_monitor
        start_log_collector
    else
        log "Not in dev mode - skipping monitoring"
    fi
}

cmd_down() {
    # Stop dev mode monitoring first
    stop_background_monitor
    stop_log_collector

    log "Stopping all services..."

    # Stop VPN services first
    for svc in "${VPN_SERVICES[@]}"; do
        if [[ -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" ]]; then
            log "Stopping VPN service: $svc..."
            docker compose -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" down 2>/dev/null || true
        fi
    done

    # Stop GPU services
    for svc in "${GPU_SERVICES[@]}"; do
        if [[ -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" ]]; then
            log "Stopping GPU service: $svc..."
            docker compose -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" down 2>/dev/null || true
        fi
    done

    for svc in "${SERVICES[@]}"; do
        log "Stopping $svc..."
        docker compose -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" down
    done
    log "All services stopped."
}

cmd_status() {
    echo ""
    echo "Service Status:"
    echo "==============="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "^(NAMES|chroma|es-|zotero|marker|redis|flower|retrieve-academic|firecrawl|playwright)" || echo "No services running"
    echo ""
}

cmd_logs() {
    local svc="${1:-}"
    if [[ -z "$svc" ]]; then
        warn "Usage: $0 logs <service>"
        warn "Services: ${SERVICES[*]}"
        exit 1
    fi
    docker compose -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" logs -f
}

cmd_backup() {
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_path="${BACKUP_DIR}/${timestamp}"
    mkdir -p "$backup_path"

    log "Creating backup at ${backup_path}..."

    # Stop services for consistent backup
    warn "Stopping services for consistent backup..."
    cmd_down 2>/dev/null || true

    # Backup bind-mounted data (chroma, zotero)
    for svc in chroma zotero; do
        if [[ -d "${SCRIPT_DIR}/${svc}/data" ]]; then
            log "Backing up ${svc} data..."
            tar -czf "${backup_path}/${svc}-data.tar.gz" -C "${SCRIPT_DIR}/${svc}" data
        fi
    done

    # Backup Docker named volumes (elasticsearch)
    for svc in elasticsearch-coherence elasticsearch-forgotten; do
        local volume_name="${svc//-/_}_esdata"
        if docker volume inspect "$volume_name" &>/dev/null; then
            log "Backing up ${svc} volume..."
            docker run --rm -v "${volume_name}:/source:ro" -v "${backup_path}:/backup" \
                alpine tar -czf "/backup/${svc}-esdata.tar.gz" -C /source .
        fi
    done

    log "Backup complete: ${backup_path}"
    ls -lh "$backup_path"

    # Restart services
    log "Restarting services..."
    cmd_up
}

cmd_restore() {
    local backup_path="${1:-}"
    if [[ -z "$backup_path" || ! -d "$backup_path" ]]; then
        warn "Usage: $0 restore <backup-path>"
        warn "Available backups:"
        ls -1 "${BACKUP_DIR}" 2>/dev/null || echo "  No backups found"
        exit 1
    fi

    warn "This will REPLACE current data with backup from: $backup_path"
    read -p "Are you sure? (yes/no): " confirm
    [[ "$confirm" != "yes" ]] && { echo "Aborted."; exit 1; }

    cmd_down 2>/dev/null || true

    # Restore bind-mounted data
    for svc in chroma zotero; do
        if [[ -f "${backup_path}/${svc}-data.tar.gz" ]]; then
            log "Restoring ${svc} data..."
            rm -rf "${SCRIPT_DIR}/${svc}/data"
            tar -xzf "${backup_path}/${svc}-data.tar.gz" -C "${SCRIPT_DIR}/${svc}"
        fi
    done

    # Restore Docker volumes
    for svc in elasticsearch-coherence elasticsearch-forgotten; do
        local volume_name="${svc//-/_}_esdata"
        if [[ -f "${backup_path}/${svc}-esdata.tar.gz" ]]; then
            log "Restoring ${svc} volume..."
            docker volume rm "$volume_name" 2>/dev/null || true
            docker volume create "$volume_name"
            docker run --rm -v "${volume_name}:/dest" -v "${backup_path}:/backup:ro" \
                alpine sh -c "tar -xzf /backup/${svc}-esdata.tar.gz -C /dest"
        fi
    done

    log "Restore complete. Starting services..."
    cmd_up
}

cmd_reset() {
    warn "This will DESTROY all service data and start fresh!"
    warn "Services: ${SERVICES[*]}"
    read -p "Type 'reset' to confirm: " confirm
    [[ "$confirm" != "reset" ]] && { echo "Aborted."; exit 1; }

    log "Stopping and removing all services..."
    for svc in "${SERVICES[@]}"; do
        docker compose -f "${SCRIPT_DIR}/${svc}/docker-compose.yml" down -v 2>/dev/null || true
    done

    # Remove bind-mounted data
    for svc in chroma zotero; do
        if [[ -d "${SCRIPT_DIR}/${svc}/data" ]]; then
            log "Removing ${svc} data..."
            rm -rf "${SCRIPT_DIR}/${svc}/data"
        fi
    done

    log "Reset complete. Run '$0 up' to start fresh."
}

cmd_monitor() {
    "$PYTHON" "${SCRIPT_DIR}/monitor.py" "${@}"
}

start_background_monitor() {
    # Only run in dev mode
    if ! is_dev_mode; then
        return 0
    fi

    # Start monitor in background with JSON output (saves to files)
    if [[ -f "$MONITOR_PID_FILE" ]]; then
        local pid=$(cat "$MONITOR_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Monitor already running (PID $pid)"
            return 0
        fi
        # Stale PID file, remove it
        rm -f "$MONITOR_PID_FILE"
    fi

    log "Starting background monitor (using $PYTHON)..."
    # Log to file so we can debug failures, truncate on each start
    nohup "$PYTHON" "${SCRIPT_DIR}/monitor.py" --json > "$MONITOR_LOG_FILE" 2>&1 &
    local monitor_pid=$!
    echo $monitor_pid > "$MONITOR_PID_FILE"

    # Give it a moment to fail if there's an import error
    sleep 1
    if kill -0 "$monitor_pid" 2>/dev/null; then
        log "Monitor started (PID $monitor_pid, log: $MONITOR_LOG_FILE)"
    else
        error "Monitor failed to start! Check $MONITOR_LOG_FILE"
        rm -f "$MONITOR_PID_FILE"
    fi
}

stop_background_monitor() {
    if [[ -f "$MONITOR_PID_FILE" ]]; then
        local pid=$(cat "$MONITOR_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping monitor (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            rm -f "$MONITOR_PID_FILE"
        else
            rm -f "$MONITOR_PID_FILE"
        fi
    fi
}

start_log_collector() {
    # Only run in dev mode
    if ! is_dev_mode; then
        return 0
    fi

    # Check if already running
    if [[ -f "$LOG_COLLECTOR_PID_FILE" ]]; then
        local pid=$(cat "$LOG_COLLECTOR_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Log collector already running (PID $pid)"
            return 0
        fi
        rm -f "$LOG_COLLECTOR_PID_FILE"
    fi

    mkdir -p "$LOGS_DIR"
    local timestamp=$(date +%Y%m%d-%H%M%S)

    log "Starting docker log collection (dev mode)..."

    # Collect logs from all running containers in background
    # Uses docker logs --follow to stream logs to files
    local pids_file="${LOG_COLLECTOR_PID_FILE}.pids"
    rm -f "$pids_file"

    for container in $(docker ps --format '{{.Names}}' 2>/dev/null); do
        local log_file="${LOGS_DIR}/${container}-${timestamp}.log"
        # Start log streaming in background for each container
        nohup docker logs -f "$container" > "$log_file" 2>&1 &
        echo $! >> "$pids_file"
    done

    # Count how many we started
    local count=$(wc -l < "$pids_file" 2>/dev/null || echo 0)
    log "Log collector started ($count containers -> $LOGS_DIR/*-${timestamp}.log)"
}

stop_log_collector() {
    # Kill all log streaming processes
    if [[ -f "${LOG_COLLECTOR_PID_FILE}.pids" ]]; then
        while read -r pid; do
            kill "$pid" 2>/dev/null || true
        done < "${LOG_COLLECTOR_PID_FILE}.pids"
        rm -f "${LOG_COLLECTOR_PID_FILE}.pids"
    fi

    if [[ -f "$LOG_COLLECTOR_PID_FILE" ]]; then
        local pid=$(cat "$LOG_COLLECTOR_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping log collector (PID $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$LOG_COLLECTOR_PID_FILE"
    fi
}

cmd_help() {
    cat <<EOF
Usage: $0 <command> [args]

Commands:
  up        Start all services
  down      Stop all services
  status    Show service status
  logs      Follow logs for a service (e.g., $0 logs zotero)
  monitor   Monitor service performance (e.g., $0 monitor --interval 60)
  backup    Create timestamped backup of all data
  restore   Restore from backup (e.g., $0 restore backups/20251216-120000)
  reset     Stop services and DELETE all data (requires confirmation)
  help      Show this help

Monitor options:
  --interval, -i  Collection interval in seconds (default: 30)
  --window, -w    Rolling stats window in seconds (default: 300)
  --json, -j      Output JSON to console instead of table
  --once          Single collection, then exit
  --no-save       Disable file persistence (console only)

Services:     ${SERVICES[*]}
GPU Services: ${GPU_SERVICES[*]} (require nvidia-container-toolkit)
VPN Services: ${VPN_SERVICES[*]} (require .env with VPN credentials)
EOF
}

# Main
case "${1:-}" in
    up)      cmd_up ;;
    down)    cmd_down ;;
    status)  cmd_status ;;
    logs)    cmd_logs "$2" ;;
    monitor) cmd_monitor "${@:2}" ;;
    backup)  cmd_backup ;;
    restore) cmd_restore "$2" ;;
    reset)   cmd_reset ;;
    help|-h|--help) cmd_help ;;
    *)       cmd_help; exit 1 ;;
esac
