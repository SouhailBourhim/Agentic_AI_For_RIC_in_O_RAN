#!/usr/bin/env bash
# Builds our gNB image: OAI v3.0.0 + kpm-per-ue.patch (M07a) + slice-control.patch (M07b), with
# OAI's own Dockerfiles.
#
# The steps are those of OAI's docker/README.md at v3.0.0 (ran-base -> ran-build -> oai-gnb),
# unchanged; the only difference from the official oai-gnb:v3.0.0 image is the patches.
# Takes a long time (UHD and all OAI targets are compiled) and about 10 GB of disk for the
# intermediate images; run it yourself, from the repo root:
#
#   testbed/gnb/build.sh            # -> agentic-ric/oai-gnb:v3.0.0-slice
#
# The M07a image (KPM patch only) was agentic-ric/oai-gnb:v3.0.0-kpmfix; it is not rebuilt.
# The OAI source goes to var/build/oai (git-ignored). Re-running re-applies the patch to a
# clean checkout, and docker's layer cache skips ran-base.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC="$REPO/var/build/oai"
PATCHES=("$REPO/testbed/gnb/kpm-per-ue.patch" "$REPO/testbed/gnb/slice-control.patch")
IMAGE=agentic-ric/oai-gnb:v3.0.0-slice

if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch v3.0.0 https://github.com/duranta-project/openairinterface5g.git "$SRC"
    git -C "$SRC" submodule update --init --depth 1 openair2/E2AP/flexric   # the E2 agent
fi
git -C "$SRC" reset -q --hard         # clean tree, then ours
git -C "$SRC" clean -fdq             # also drop files a previous patch added (the flexric submodule stays)
for patch in "${PATCHES[@]}"; do
    git -C "$SRC" apply "$patch"
done
echo "patches applied: $(git -C "$SRC" status --short | wc -l) files changed or added"

cd "$SRC"
docker build --target ran-base  --tag ran-base:latest  --file docker/Dockerfile.base.ubuntu .
docker build --target ran-build --tag ran-build:latest --file docker/Dockerfile.build.ubuntu .
docker build --target oai-gnb   --tag "$IMAGE"         --file docker/Dockerfile.gNB.ubuntu .
docker image ls "$IMAGE"
