# The RL xApp (M07b): the RL runner + the registry's champion policy + the slice-control xApp.
#
# One container = the "xApp RL" box of figure 2. Inside it the runner (Python) talks to
# xapp_slice_ctrl (C, FlexRIC SDK) through a pipe; xapp_slice_ctrl talks to the RIC over E42.
# Built by xapps/rl/build.sh, never by hand: the script prepares the build context (the code, the
# champion's SB3 file, uv.lock as constraints) and puts the model version in the image labels.
#
# Builder stage: the build steps of FlexRIC's docker/Dockerfile.flexric.ubuntu at v3.0.0, as for
# the KPM xApp (xapps/kpm/Dockerfile), plus one small patch to the xApp library, building only
# xapp_slice_ctrl. Runtime stage: the official
# image, so the service-model libraries match the RIC's, plus Python 3.12 from uv.

ARG FLEXRIC_TAG=v3.0.0

FROM ubuntu:jammy AS builder
ARG FLEXRIC_TAG
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --yes --no-install-recommends \
       build-essential ca-certificates libsctp-dev git m4 automake libtool python3 cmake \
       bison flex libpcre2-dev gcc-12 g++-12 ninja-build && \
    apt-get clean
RUN git init -q /tmp/asn1c && cd /tmp/asn1c && \
    git fetch --depth 1 https://github.com/mouse07410/asn1c 940dd5fa9f3917913fd487b13dfddfacd0ded06e && \
    git checkout -q FETCH_HEAD && \
    autoreconf -iv && ./configure --prefix /opt/asn1c/ && make -j"$(nproc)" && make install
RUN git clone --depth 1 --branch ${FLEXRIC_TAG} https://github.com/duranta-project/flexric.git /flexric
# Fix a race in the xApp library: register the control's timeout before sending it, so a fast
# acknowledge cannot arrive first and crash the xApp (xapps/slice_ctrl/flexric-control-ack-race.patch).
COPY slice_ctrl/flexric-control-ack-race.patch /tmp/
RUN git -C /flexric apply /tmp/flexric-control-ack-race.patch
COPY slice_ctrl/xapp_slice_ctrl.c /flexric/examples/xApp/c/slice/
COPY slice_ctrl/CMakeLists.append.txt /tmp/
RUN cat /tmp/CMakeLists.append.txt >> /flexric/examples/xApp/c/slice/CMakeLists.txt && \
    mkdir /flexric/build && cd /flexric/build && \
    cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DE2AP_VERSION=E2AP_V3 -DKPM_VERSION=KPM_V3_00 \
          -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 .. && \
    ninja xapp_slice_ctrl

FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM oaisoftwarealliance/oai-flexric:${FLEXRIC_TAG}
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 VIRTUAL_ENV=/opt/venv
RUN uv venv /opt/venv --python 3.12
COPY requirements.txt constraints.txt /tmp/
RUN uv pip install -r /tmp/requirements.txt -c /tmp/constraints.txt \
       --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
       --index-strategy unsafe-best-match
COPY --from=builder /flexric/build/examples/xApp/c/slice/xapp_slice_ctrl /usr/local/flexric/xApp/c/slice/
COPY code/ /app/
COPY model/ /app/model/

ARG MODEL_FILE
ARG MODEL_LABEL
ARG MODEL_VERSION
LABEL org.agentic-ric.model.name="slice-policy" \
      org.agentic-ric.model.version="${MODEL_VERSION}" \
      org.agentic-ric.model.label="${MODEL_LABEL}" \
      org.agentic-ric.model.file="${MODEL_FILE}"
ENV MODEL_FILE=${MODEL_FILE} MODEL_LABEL=${MODEL_LABEL} PYTHONUNBUFFERED=1
WORKDIR /app
# var/ is bind-mounted from the repo: var/ric-testbed.db and var/events.jsonl, shared with the
# KPM monitor, the agent and the exporter running in WSL.
CMD ["sh", "-c", "exec /opt/venv/bin/python -m rl.runner --backend flexric --keep-db --policy-file /app/model/$MODEL_FILE --model-label \"$MODEL_LABEL [image]\" $RUNNER_ARGS"]
