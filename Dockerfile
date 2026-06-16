# FEniCSx finite-element tutorial — development container.
#
# Built on the upstream DOLFINx image, which already ships the FEniCSx stack
# (dolfinx, ufl, basix, ffcx), PETSc, MPI, Python 3.12 and NumPy. We add the
# visualization libraries, the dolfinx_mpc extension (periodic BCs for Example
# 3), the Claude Code / Gemini CLIs, and an unprivileged user for the VSCode
# dev-container flow.
#
# Pinned to a specific DOLFINx release (rather than the floating `:stable` tag)
# so the from-source dolfinx_mpc build below stays version-matched and the image
# is reproducible.
FROM dolfinx/dolfinx:v0.11.0

# 1. Small dev / CLI conveniences (the scientific stack is already in the base
#    image): git + curl for VCS/downloads, tmux for agent sessions, jq for
#    JSON, and 256-colour terminal + UTF-8 locale support.
RUN apt-get update && apt-get install -y \
    git \
    curl \
    tmux \
    jq \
    ncurses-term \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 2. Visualization + animation packages used by the plotting scripts. NumPy
#    and SciPy already ship in the base image; matplotlib renders the
#    convergence and contour plots, imageio assembles the cavity animation.
RUN pip3 install --no-cache-dir --break-system-packages \
    matplotlib \
    scipy \
    imageio

# 2b. dolfinx_mpc — the multi-point-constraint extension that provides periodic
#     boundary conditions, used by Example 3 (Rayleigh-Darcy convection). It is
#     not in the base image and not on PyPI, so we build it from a pinned commit
#     against the image's DOLFINx. The base image already exports
#     CMAKE_PREFIX_PATH / LD_LIBRARY_PATH for /usr/local/dolfinx-real, so the C++
#     library installs alongside libdolfinx and the Python package links to it.
ARG DOLFINX_MPC_COMMIT=851914c3715f47af335a9e61a2433af532a4a28b
RUN git clone https://github.com/jorgensd/dolfinx_mpc.git /tmp/dolfinx_mpc \
    && git -C /tmp/dolfinx_mpc checkout ${DOLFINX_MPC_COMMIT} \
    && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_PREFIX_PATH=/usr/local/dolfinx-real \
       -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real \
       -B /tmp/dolfinx_mpc/build-dir /tmp/dolfinx_mpc/cpp \
    && ninja -C /tmp/dolfinx_mpc/build-dir install \
    && pip3 install --no-cache-dir --break-system-packages --no-build-isolation \
       --config-settings=cmake.build-type=Release \
       --config-settings=cmake.define.CMAKE_PREFIX_PATH=/usr/local/dolfinx-real \
       /tmp/dolfinx_mpc/python \
    && rm -rf /tmp/dolfinx_mpc

# 3. en_US.UTF-8 locale for proper Unicode rendering in the terminal.
RUN sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen

# 4. Unprivileged user for the dev container (UID/GID configurable at build
#    time). Ubuntu 24.04 ships a default 'ubuntu' user at UID 1000 — remove it
#    first so 'demo' can claim 1000.
ARG UID=1000
ARG GID=1000
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && (groupdel ubuntu 2>/dev/null || true) \
    && groupadd -g ${GID} demo \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash demo \
    && chown -R demo:demo /home/demo

# Placeholder git identity so the agent's commits succeed out of the box (git
# refuses to commit without one). Override with
# `git config --global user.{name,email} ...` for your own.
RUN git config --system user.name "FEniCS FE Demo" \
    && git config --system user.email "demo@example.com"

# Terminal + locale environment.
ENV TERM=xterm-256color \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    COLORTERM=truecolor

# 5. Node.js — required by the Claude Code + Gemini CLIs.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 6. Claude Code + Gemini CLI, installed globally.
RUN npm install -g @anthropic-ai/claude-code @google/gemini-cli

# 7. Working directory (matches the dev container's workspace mount).
WORKDIR /demo

# 8. Default to an interactive shell. The dev container keeps itself alive via
#    overrideCommand and attaches as the `demo` user (remoteUser).
CMD ["bash"]
