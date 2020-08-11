FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7                                                                      
ENV TZ Asia/Seoul
ARG  DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -q \
        build-essential \
        pkg-config \
        software-properties-common \
        curl \
        git \
        unzip \
        zlib1g-dev \
        locales \
    && apt-get clean -qq && rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev git-all


RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US.en LC_ALL=en_US.UTF-8
ENV PYTHONVERSION=3.6.9

RUN git clone https://github.com/NVIDIA/apex \
 && cd apex \
 && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./                                                                                                                                                
RUN pip install cython
RUN git clone https://github.com/dwgoon/jpegio.git \
 && cd jpegio \
 && python setup.py install

RUN pip install opencv-python glob3 scikit-learn Pillow pandas tensorflow==1.13.1 torch_dct fire shapely Cython scipy pandas pyyaml json_tricks scikit-image yacs>=0.1.5 tensorboardX>=1.6

CMD ["/bin/bash"]

