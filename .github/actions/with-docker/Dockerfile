ARG K_VERSION
FROM runtimeverificationinc/kframework-k:ubuntu-jammy-${K_VERSION}

SHELL ["/bin/bash", "-c"]

RUN    apt-get update        \
    && apt-get upgrade --yes \
    && apt-get install --yes \
           curl              \
           jq                \
    && apt-get clean

ARG USER=user
ARG GROUP=$USER
ARG USER_ID=1000
ARG GROUP_ID=$USER_ID

RUN groupadd -g $GROUP_ID $GROUP && useradd -m -u $USER_ID -s /bin/sh -g $GROUP $USER

USER $USER:$GROUP

ENV PATH=/home/$USER/.local/bin:$PATH

ARG UV2NIX_REVISION
RUN UV_VERSION=$(curl -s https://raw.githubusercontent.com/pyproject-nix/uv2nix/${UV2NIX_REVISION}/pkgs/uv-bin/srcs.json | jq -r .version) \
    && [[ "$UV_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    && curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh && uv --version