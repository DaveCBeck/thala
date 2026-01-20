#!/usr/bin/env bash
# Reset script - clears caches, logs, test data, and optionally permanent stores
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="${PROJECT_DIR}/services"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()   { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; }
info()  { echo -e "${BLUE}[i]${NC} $1"; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Clears caches, logs, test data, and __pycache__ directories.

Options:
  --full, -f    Also reset permanent stores (chroma, zotero, elasticsearch)
                This calls 'services/services.sh reset' and requires confirmation.
  --dry-run     Show what would be deleted without actually deleting
  --help, -h    Show this help

What gets cleared (always):
  - .cache/*           (embeddings, marker, openalex, test caches)
  - logs/              (workflow logs)
  - testing/test_data  (test output files)
  - testing/traces     (trace files)
  - **/__pycache__     (Python bytecode caches)

What gets cleared with --full:
  - services/chroma/data     (vector store)
  - services/zotero/data     (reference manager)
  - elasticsearch volumes    (coherence, forgotten indices)
EOF
}

# Parse arguments
FULL_RESET=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full|-f)
            FULL_RESET=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Helper to delete directory contents (keeps the directory)
empty_dir() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        local count=$(find "$dir" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
        if [[ "$count" -gt 0 ]]; then
            if $DRY_RUN; then
                info "Would empty: $dir ($count items)"
            else
                rm -rf "${dir:?}"/*
                log "Emptied: $dir ($count items)"
            fi
        else
            info "Already empty: $dir"
        fi
    else
        info "Does not exist: $dir"
    fi
}

# Main cleanup
echo ""
info "Thala Reset Script"
info "=================="
echo ""

# 1. Empty .cache subdirectories
log "Clearing cache directories..."
for cache_dir in "${PROJECT_DIR}/.cache"/*; do
    if [[ -d "$cache_dir" ]]; then
        empty_dir "$cache_dir"
    fi
done

# 2. Empty logs directory
log "Clearing logs..."
empty_dir "${PROJECT_DIR}/logs"

# 3. Empty testing directories
log "Clearing test data..."
empty_dir "${PROJECT_DIR}/testing/test_data"
empty_dir "${PROJECT_DIR}/testing/traces"

# 4. Delete __pycache__ directories
log "Removing __pycache__ directories..."
if $DRY_RUN; then
    pycache_count=$(find "$PROJECT_DIR" -type d -name "__pycache__" 2>/dev/null | wc -l)
    info "Would remove: $pycache_count __pycache__ directories"
else
    pycache_count=$(find "$PROJECT_DIR" -type d -name "__pycache__" 2>/dev/null | wc -l)
    find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    log "Removed: $pycache_count __pycache__ directories"
fi

# 5. Optional: Full reset of permanent stores
if $FULL_RESET; then
    echo ""
    warn "========================================="
    warn "  FULL RESET - Permanent Data Deletion  "
    warn "========================================="
    warn ""
    warn "This will DESTROY:"
    warn "  - Chroma vector store (services/chroma/data)"
    warn "  - Zotero library (services/zotero/data)"
    warn "  - Elasticsearch indices (coherence, forgotten)"
    warn ""

    if $DRY_RUN; then
        info "Dry run: Would call 'services/services.sh reset'"
    else
        read -p "Type 'reset' to confirm permanent data deletion: " confirm
        if [[ "$confirm" != "reset" ]]; then
            warn "Aborted. Temporary files were still cleared."
            exit 0
        fi

        echo ""
        log "Running services.sh reset..."

        # Call the services reset script (it has its own confirmation, but we already got ours)
        # We need to skip its confirmation since we already confirmed
        # Replicate the relevant parts directly:

        log "Stopping and removing all services..."
        for svc in elasticsearch-coherence elasticsearch-forgotten chroma zotero firecrawl; do
            if [[ -f "${SERVICES_DIR}/${svc}/docker-compose.yml" ]]; then
                docker compose -f "${SERVICES_DIR}/${svc}/docker-compose.yml" down -v 2>/dev/null || true
            fi
        done

        # Remove bind-mounted data
        for svc in chroma zotero; do
            if [[ -d "${SERVICES_DIR}/${svc}/data" ]]; then
                log "Removing ${svc} data..."
                rm -rf "${SERVICES_DIR}/${svc}/data"
            fi
        done

        log "Permanent stores reset complete."
    fi
fi

echo ""
log "Reset complete!"
if $FULL_RESET && ! $DRY_RUN; then
    info "Run 'services/services.sh up' to start fresh services."
fi
