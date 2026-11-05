#!/usr/bin/env bash
# Builds the RL xApp image with the MLflow registry's current champion inside (M07b).
#
#   xapps/rl/build.sh                     # -> agentic-ric/rl-xapp:v<champion version>, also tagged :latest
#
# The build context is assembled in var/build/rl-xapp-context (git-ignored), with only what the
# runner needs: the Python packages it imports, the C xApp sources, the champion's SB3 file and
# uv.lock exported as pip constraints, so the image runs the same versions as the tests.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
CTX="$REPO/var/build/rl-xapp-context"
cd "$REPO"

# Champion: model name, version and local file, from the registry (as the runner would load it).
read -r MODEL_VERSION MODEL_NAME MODEL_PATH < <(uv run python - <<'PY'
from mlops import registry
found = registry.champion()
version = registry.alias_version(registry.CHAMPION)
if found is None or version is None:
    raise SystemExit("no champion in the MLflow registry")
print(version.version, found[0], found[1])
PY
)
MODEL_FILE=$(basename "$MODEL_PATH")
MODEL_LABEL="slice-policy v$MODEL_VERSION ($MODEL_NAME)"
IMAGE="agentic-ric/rl-xapp:v$MODEL_VERSION"
echo "champion: $MODEL_LABEL, file $MODEL_FILE"

rm -rf "$CTX"
mkdir -p "$CTX/code/xapps" "$CTX/model" "$CTX/slice_ctrl"
cp -r rl sim store backends verifier "$CTX/code/"
cp xapps/__init__.py xapps/kpm_monitor.py "$CTX/code/xapps/"
find "$CTX/code" -name __pycache__ -prune -exec rm -rf {} +
cp xapps/slice_ctrl/xapp_slice_ctrl.c xapps/slice_ctrl/CMakeLists.append.txt \
   xapps/slice_ctrl/flexric-control-ack-race.patch "$CTX/slice_ctrl/"
cp "$MODEL_PATH" "$CTX/model/$MODEL_FILE"
cp xapps/rl/Dockerfile xapps/rl/requirements.txt "$CTX/"
uv export --no-dev --no-hashes --no-emit-project --quiet > "$CTX/constraints.txt"

docker build --tag "$IMAGE" --tag agentic-ric/rl-xapp:latest \
    --build-arg MODEL_FILE="$MODEL_FILE" --build-arg MODEL_LABEL="$MODEL_LABEL" \
    --build-arg MODEL_VERSION="$MODEL_VERSION" "$CTX"
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}'
