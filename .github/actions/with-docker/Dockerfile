ARG K_VERSION
FROM runtimeverificationinc/kframework-k:ubuntu-jammy-${K_VERSION}

RUN    apt-get update        \
    && apt-get upgrade --yes \
    && apt-get install --yes \
           curl              \
    && apt-get clean

ARG USER=user
ARG GROUP=$USER
ARG USER_ID=1000
ARG GROUP_ID=$USER_ID

RUN groupadd -g $GROUP_ID $GROUP && useradd -m -u $USER_ID -s /bin/sh -g $GROUP $USER

USER $USER:$GROUP

ENV PATH=/home/$USER/.local/bin:$PATH
RUN curl -sSL https://install.python-poetry.org | python3 -
