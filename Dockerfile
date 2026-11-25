# syntax=docker/dockerfile:1.7
# ISR-Forge headless toolchain: Python 3.11 + OpenVSP/VSPAERO + ArduPilot SITL + JSBSim.
# Every C++ dependency is built from source so the image works on amd64 and arm64 (Apple Silicon)
# and the OpenVSP Python bindings match the Python 3.11 interpreter.

ARG OPENVSP_TAG=OpenVSP_3.56.0
ARG ARDUPILOT_TAG=Plane-4.7.2
ARG JSBSIM_TAG=v1.3.3

# ---------------------------------------------------------------- base ---
FROM ubuntu:22.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates gnupg \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv \
      git cmake build-essential pkg-config curl \
      libgl1-mesa-glx libglu1-mesa xvfb \
 && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /opt/venv
# UTF-8 locale: CMake's archive extractor fails on UTF-8 filenames in OpenVSP's bundled zips otherwise.
ENV PATH=/opt/venv/bin:$PATH \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Ubuntu 22.04 ships CMake 3.22; OpenVSP 3.56 needs >= 3.24. The PyPI wheel lands in the venv,
# which is first on PATH, so it shadows the apt cmake for every later stage.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools "cmake>=3.28,<4"

# ------------------------------------------------------ OpenVSP (source) ---
FROM base AS openvsp-build
ARG OPENVSP_TAG
ARG DEBIAN_FRONTEND=noninteractive
# FLTK is built from OpenVSP's bundled source: 3.56 links `fltk::gl`, which Ubuntu 22.04's FLTK 1.3
# does not export. The X11/GL headers below are what the bundled FLTK needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libxml2-dev libcpptest-dev libjpeg-dev libglm-dev libeigen3-dev libcminpack-dev libglew-dev \
      libx11-dev libxext-dev libxft-dev libxinerama-dev libxcursor-dev libxrender-dev libxfixes-dev \
      libgl1-mesa-dev libglu1-mesa-dev swig unzip \
 && rm -rf /var/lib/apt/lists/*
# NumPy must be importable at configure time or OpenVSP silently skips the Python API.
RUN pip install --no-cache-dir numpy
# Bundled static libs (FLTK etc.) get linked into the Python API's shared objects; on aarch64 that
# fails with R_AARCH64_ADR_PREL_PG_HI21 relocation errors unless they are position-independent.
ENV CFLAGS="-fPIC" CXXFLAGS="-fPIC"
RUN git clone --depth 1 --branch ${OPENVSP_TAG} https://github.com/OpenVSP/OpenVSP.git /src/OpenVSP
WORKDIR /build/libs
RUN cmake -DCMAKE_BUILD_TYPE=Release \
      -DVSP_USE_SYSTEM_CPPTEST=true -DVSP_USE_SYSTEM_LIBXML2=true -DVSP_USE_SYSTEM_FLTK=false \
      -DVSP_USE_SYSTEM_GLM=true -DVSP_USE_SYSTEM_GLEW=true -DVSP_USE_SYSTEM_CMINPACK=true \
      -DVSP_USE_SYSTEM_EIGEN=false -DVSP_USE_SYSTEM_LIBIGES=false -DVSP_USE_SYSTEM_EXPRPARSE=false \
      /src/OpenVSP/Libraries \
 && make -j"$(nproc)"
WORKDIR /build/vsp
RUN cmake -DCMAKE_BUILD_TYPE=Release -DVSP_LIBRARY_PATH=/build/libs \
      -DPYTHON_EXECUTABLE=/opt/venv/bin/python -DPython3_EXECUTABLE=/opt/venv/bin/python \
      /src/OpenVSP/src \
 && make -j"$(nproc)" \
 && make package \
 && mkdir -p /opt/openvsp \
 && unzip -q OpenVSP-*.zip -d /tmp/pkg \
 && cp -r /tmp/pkg/OpenVSP-*/* /opt/openvsp/

# ------------------------------------------------ ArduPilot SITL (source) ---
FROM base AS ardupilot-build
ARG ARDUPILOT_TAG
RUN pip install --no-cache-dir "empy==3.3.4" future lxml pexpect pymavlink
RUN git clone --depth 1 --branch ${ARDUPILOT_TAG} --recurse-submodules --shallow-submodules \
      https://github.com/ArduPilot/ardupilot.git /src/ardupilot
WORKDIR /src/ardupilot
RUN ./waf configure --board sitl && ./waf plane

# ------------------------------------------------------- JSBSim (source) ---
FROM base AS jsbsim-build
ARG JSBSIM_TAG
RUN git clone --depth 1 --branch ${JSBSIM_TAG} https://github.com/JSBSim-Team/jsbsim.git /src/jsbsim
WORKDIR /build/jsbsim
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MODULE=OFF -DBUILD_DOCS=OFF /src/jsbsim \
 && make -j"$(nproc)" JSBSim

# --------------------------------------------------------------- runtime ---
FROM base AS runtime
ARG DEBIAN_FRONTEND=noninteractive
# Runtime (non -dev) counterparts of the libraries OpenVSP was compiled against.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcminpack1 libxml2 libglew2.2 libjpeg8 libgomp1 libfontconfig1 \
      libx11-6 libxext6 libxft2 libxinerama1 libxcursor1 libxrender1 libxfixes3 libgl1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=openvsp-build /opt/openvsp /opt/openvsp
COPY --from=ardupilot-build /src/ardupilot/build/sitl/bin/arduplane /usr/local/bin/arduplane
COPY --from=ardupilot-build /src/ardupilot/Tools/autotest /opt/ardupilot/Tools/autotest
COPY --from=jsbsim-build /build/jsbsim/src/JSBSim /usr/local/bin/JSBSim
ENV PATH=/opt/openvsp:$PATH \
    ARDUPILOT_HOME=/opt/ardupilot

# OpenVSP Python API (bindings compiled above against this exact interpreter). The local packages
# depend on each other by name, so they go in one pip call to resolve locally rather than from PyPI.
# CHARM / AvlPy / pyPMARC are unrelated solvers and are skipped.
RUN cd /opt/openvsp/python \
 && pip install --no-cache-dir ./vsp_airfoils ./utilities ./degen_geom ./openvsp_config ./openvsp

WORKDIR /app
COPY pyproject.toml ./
# Dependencies first, from pyproject.toml alone, with a persistent pip cache: editing src/ no longer
# invalidates this layer, and changing pyproject reuses already-downloaded wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src && touch src/__init__.py \
 && pip install ".[physics,agent,dev,ui,report]" \
 && pip uninstall -y isr-forge && rm -rf src build ./*.egg-info
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && rm -rf build ./*.egg-info
COPY tests ./tests
COPY data/airfoils ./data/airfoils
COPY .streamlit ./.streamlit
# Regenerate the catalog from source so the image never ships a stale or missing DB.
RUN python -m src.tools.database.seed_data

# Fail the build loudly if any native toolchain piece or shared library is missing.
RUN for bin in /opt/openvsp/vspaero /usr/local/bin/arduplane /usr/local/bin/JSBSim \
               $(find /opt/venv/lib -name '_vsp*.so'); do \
      if ldd "$bin" | grep "not found"; then echo "missing libs for $bin" && exit 1; fi; \
    done \
 && python -c "import openvsp, jsbsim, pymavlink, build123d, aerosandbox" \
 && command -v vspaero && command -v arduplane && command -v JSBSim

# Propeller CT/CP tables (spec 6.1): built once here so the first sizing run in a container starts warm. They live
# outside /app, so `docker run -v "$PWD":/app` and the compose volumes never shadow them, and the image's own source
# always matches its tables (a source mount with a different grid builds the missing ones on demand, in the
# container only).
ENV PROP_TABLE_CACHE_DIR=/opt/prop_tables
RUN python -m src.tools.propulsion.prop_tables --warm

RUN useradd --create-home --uid 1000 forge && chown -R forge:forge /app "$PROP_TABLE_CACHE_DIR"
USER forge

# OpenVSP/VSPAERO run headless through the Python API, so no X server is needed. (xvfb-run is
# still installed for GUI tooling, but wrapping pytest in it hangs after the run completes.)
CMD ["pytest", "-q", "-p", "no:cacheprovider"]
