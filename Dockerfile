#   
# Thermal Segmentor - A Semi-Supervised Thermal Segmentation Algorithm using [D]eep Learning approach
# Supervisor: Professor Xavier Maldague
# Student: Parham Nooralishahi, PhD. student @ Computer and Electrical Engineering Department, Université Laval
# University: Université Laval

FROM pytorch/pytorch:latest

# Define the metadata labels
LABEL author1="Parham Nooralishahi"
LABEL email1="parham.nooralishahi.1@ulaval.ca"
LABEL superviser="Professor Xavier Maldague"
LABEL organization="Laval University"

ARG DEBIAN_FRONTEND=noninteractive
ARG RUN_SCRIPT=run_program.py

ENV TZ=America/Montreal

RUN apt-get update && \
    apt-get install -y apt-transport-https 

RUN apt-get install -y git wget zip build-essential cmake sysvbanner

RUN echo "" && \
    banner UNIVERSITE LAVAL && \
    echo "-----------------------------------------"

# Install Dependencies
RUN pip install pytorch-ignite torchmetrics torchvision comet-ml opencv-python scikit-image scikit-learn dotmap phasepack cython pyyaml matplotlib
RUN pip install crfseg tqdm pyfftw dotmap
RUN pip install git+https://github.com/waspinator/coco.git@2.1.0
RUN pip install git+https://github.com/waspinator/pycococreator.git@0.2.0
RUN pip install gimp-labeling-converter
RUN pip install segmentation-models-pytorch

# RUN pip install pytorch-lightning comet-ml scikit-image opencv-python 

RUN mkdir -p /phm

COPY . /phm
WORKDIR /phm