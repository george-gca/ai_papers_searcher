# Created by: George Corrêa de Araújo (george.gcac@gmail.com)
# ==================================================================

# FROM python:latest
FROM python:3.11

ARG GROUPID=901
ARG GROUPNAME=searcher
ARG USERID=901
ARG USERNAME=user

# Environment variables

RUN PIP_INSTALL="pip --no-cache-dir install --upgrade" && \

# ==================================================================
# Create a system group with name deeplearning and id 901 to avoid
#    conflict with existing uids on the host system
# Create a system user with id 901 that belongs to group deeplearning
# ------------------------------------------------------------------

    groupadd -r $GROUPNAME -g $GROUPID && \
    # useradd -u $USERID -r -g $GROUPNAME $USERNAME && \
    useradd -u $USERID -m -g $GROUPNAME $USERNAME && \

# ==================================================================
# python libraries via pip
# ------------------------------------------------------------------

    $PIP_INSTALL \
        pip \
        wheel && \
    $PIP_INSTALL \
        flask \
        flask-paginate \
        gunicorn \
        numpy \
        pandas \
        pyarrow \
        scipy \
        unidecode && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# Flask port
EXPOSE 5000

USER $USERNAME
