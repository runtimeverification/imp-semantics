ARG K_VERSION

FROM runtimeverificationinc/kframework-k:ubuntu-jammy-$K_VERSION

ARG K_VERSION

# create non-root user and adjust UID:GID on start-up
# see https://github.com/boxboat/fixuid
RUN addgroup --gid 1000 k-group && \
    adduser -uid 1000 --ingroup k-group --home /home/k-user --shell /bin/sh --disabled-password --gecos "" k-user
RUN apt-get install -y curl graphviz python-is-python3 && \
    USER=k-user && \
    GROUP=k-group && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER k-user:k-group
WORKDIR /home/k-user
ENV K_VERSION=${K_VERSION} \
    PATH=/home/k-user/.local/bin:$PATH \
    force_color_prompt=yes

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# install kimp
COPY --chown=k-user:k-group . /home/k-user/kimp-src
RUN poetry -C /home/k-user/kimp-src build && \
    pip install /home/k-user/kimp-src/dist/*.whl && \
    rm -r /home/k-user/kimp-src && \
    kdist --verbose build -j3

WORKDIR /home/k-user/workspace
COPY --chown=k-user:k-group examples examples
COPY --chown=k-user:k-group  ./package/smoke-test.sh .

ENTRYPOINT ["fixuid", "-q"]

CMD ["printf", "%s\n",\
     "Welcome to the K Framework!",\
     "",\
     "This docker image provides a K-framework installation with the following programs:",\
     " * kompile",\
     " * krun",\
     " * kprove",\
     " * kast",\
     " * K backend tools (kore-*)",\
     "",\
     "as well as a pre-installed pyk library to interact with K programmatically.",\
     "",\
     "To use this docker image, start a container with an interactive shell and",\
     "a working directory with your K definition mounted into it, like so:",\
     "",\
     "user@host$ docker run --rm -it -v \"$PWD\":/home/k-user/workspace -u $(id -u):$(id -g) <docker-image> /bin/bash",\
     ""]
