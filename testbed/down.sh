#!/usr/bin/env bash
# Stop the testbed in reverse order: RAN side first (xApps, UEs, gNB, RIC: compose stops the
# services that depend on others first), then the core, which owns the shared network.
#
# Usage: testbed/down.sh

set -euo pipefail
cd "$(dirname "$0")"

docker compose -f rfsim-flexric/docker-compose.yml down
docker compose -f oai-cn5g/docker-compose.yaml down
