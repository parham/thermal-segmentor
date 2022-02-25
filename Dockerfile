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

# Install Pytorch-Lightening
RUN pip install pytorch-lightning
# Install Comet
RUN pip install comet-ml scikit-image

RUN mkdir -p /phm

COPY . /phm
WORKDIR /phm
