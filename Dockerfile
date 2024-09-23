FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    build-essential \
    libhdf5-serial-dev \
    hdf5-tools \
    python3-dev
RUN conda install scipy numpy matplotlib numba
RUN conda install pytorch torchvision -c pytorch
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding pyaml igraph ujson
RUN conda install pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
COPY . .
CMD /bin/bash
