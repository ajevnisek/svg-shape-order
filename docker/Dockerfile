ARG UBUNTU_VER=16.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64


FROM nvidia/cuda:11.0.3-base-ubuntu${UBUNTU_VER}
RUN apt update
RUN apt install python3 python3-dev python3-pip wget build-essential git -y
RUN python3 -m pip install --upgrade pip
RUN apt-get clean -y
RUN rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
#WORKDIR /diffusion
RUN git clone https://github.com/ajevnisek/svg-shape-order.git
WORKDIR svg-shape-order
COPY environment.yaml .
RUN conda env create -f environment.yaml
SHELL ["conda", "run", "-n", "svg", "/bin/bash", "-c"]
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "svg", "python", "single_image_ellipse_parameters_regressor.py", "--config configs/single_image_3_ellipses.yaml"]
# python single_image_ellipse_parameters_regressor.py --config configs/single_image_3_ellipses.yaml
# Make RUN commands use the new environment:
#RUN echo "conda activate svg" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]
#ENTRYPOINT ["sh", "./entrypoint.sh"]
