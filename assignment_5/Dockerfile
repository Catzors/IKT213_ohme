FROM ubuntu:latest
LABEL authors="catz"

# Update once and install everything at the same time
RUN apt-get update -y && \
    apt-get install -y --fix-missing \
        build-essential \
        cmake \
        gfortran \
        git \
        wget \
        curl \
        graphicsmagick \
        libgraphicsmagick1-dev \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libgtk2.0-dev \
        libjpeg-dev \
        liblapack-dev \
        libswscale-dev \
        pkg-config \
        python3-dev \
        python3-numpy \
        python3-setuptools \
        software-properties-common \
        zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Build old dlib
RUN cd /root && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib && \
    cd dlib && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

ENTRYPOINT ["top", "-b"]
