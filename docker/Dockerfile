FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
WORKDIR /home
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.6 \
    python3-pip \
    python3.6-dev \
    git \
    cmake \
    vim-gnome \
    curl \
    tmux \
    sudo \
    ssh \
    zlib1g-dev \
    libbz2-dev \
    libssl-dev \
    libreadline6 \
    libreadline6-dev \
    libsqlite3-dev \
    libjpeg-dev 
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3
RUN export CUDA_HOME=/usr/local/cuda-9.2
RUN pip3 install --upgrade pip
RUN pip3 install Cython
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/kakusikun/dotfiles.git -b vim-airline --single-branch
RUN git clone https://github.com/kakusikun/deep-learning-works.git
RUN chmod -R a+w .

