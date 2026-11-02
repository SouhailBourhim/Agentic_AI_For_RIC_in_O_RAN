# The M07a KPM xApp (xapp_kpm_slices.c), built against FlexRIC v3.0.0.
#
# The official oai-flexric image has the xApp binaries but no SDK (headers, libe42_xapp), so the
# xApp is built from source. Builder stage: the build steps of FlexRIC's own
# docker/Dockerfile.flexric.ubuntu at tag v3.0.0 (Ubuntu 22.04, gcc-12, asn1c at the same
# commit, E2AP_V3 + KPM_V3_00), without SWIG (no Python bindings needed), and building only our
# target. Runtime stage: the official image itself, so the service-model libraries the xApp loads
# are exactly those of the RIC it talks to.
#
#   docker build -t agentic-ric/xapp-kpm-slices:v3.0.0 xapps/kpm

ARG FLEXRIC_TAG=v3.0.0

FROM ubuntu:jammy AS builder
ARG FLEXRIC_TAG
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --yes --no-install-recommends \
       build-essential ca-certificates libsctp-dev git m4 automake libtool python3 cmake \
       bison flex libpcre2-dev gcc-12 g++-12 ninja-build && \
    apt-get clean
# Only the pinned commit is fetched (the full history is large, and a flaky link drops it).
RUN git init -q /tmp/asn1c && cd /tmp/asn1c && \
    git fetch --depth 1 https://github.com/mouse07410/asn1c 940dd5fa9f3917913fd487b13dfddfacd0ded06e && \
    git checkout -q FETCH_HEAD && \
    autoreconf -iv && ./configure --prefix /opt/asn1c/ && make -j"$(nproc)" && make install
RUN git clone --depth 1 --branch ${FLEXRIC_TAG} https://github.com/duranta-project/flexric.git /flexric
COPY xapp_kpm_slices.c /flexric/examples/xApp/c/monitor/
COPY CMakeLists.append.txt /tmp/
RUN cat /tmp/CMakeLists.append.txt >> /flexric/examples/xApp/c/monitor/CMakeLists.txt && \
    mkdir /flexric/build && cd /flexric/build && \
    cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DE2AP_VERSION=E2AP_V3 -DKPM_VERSION=KPM_V3_00 \
          -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 .. && \
    ninja xapp_kpm_slices

FROM oaisoftwarealliance/oai-flexric:${FLEXRIC_TAG}
COPY --from=builder /flexric/build/examples/xApp/c/monitor/xapp_kpm_slices /usr/local/flexric/xApp/c/monitor/
CMD ["stdbuf", "-o0", "/usr/local/flexric/xApp/c/monitor/xapp_kpm_slices"]
