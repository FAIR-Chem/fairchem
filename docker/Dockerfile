FROM nvcr.io/nvidia/pytorch:20.02-py3
SHELL ["/bin/bash", "-c"]

# Set up a non-root user, `user`, with a group, `group`
ENV HOME=/home
RUN mkdir -p $HOME
RUN groupadd -r group && \
    useradd -r -g group -d $HOME -s /sbin/nologin -c "Default user" user
RUN cp /root/.bashrc $HOME

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh --directory-prefix=$HOME
RUN /bin/bash $HOME/Miniconda3-py37_4.8.2-Linux-x86_64.sh -bp /miniconda3
RUN rm $HOME/Miniconda3-py37_4.8.2-Linux-x86_64.sh
ENV PATH /miniconda3/bin:$PATH

# Install conda packages
RUN conda config --prepend channels conda-forge
RUN conda config --prepend channels pytorch
RUN conda install \
    cudatoolkit=10.0 \
    pytorch==1.4.0 \
    tensorboard==2.1.0 \
    pyyaml==5.2 \
    pre-commit \
    tqdm>=4.41 \
    jupyter>=1.0.0 \
    ase==3.19.0 \
    pymatgen==2020.1.10 \
    matplotlib \
    gpytorch
RUN conda clean -ity

# Install pip packages
RUN pip install --upgrade pip
RUN pip install -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    torch-scatter==latest+cu100 \
    torch-sparse==latest+cu100 \
    torch-cluster==latest+cu100 \
    torch-spline-conv==latest+cu100
RUN pip install \
    torch-geometric \
    demjson \
    wandb \
    scikit-image==0.16.2 \
    ray

# Only the development version of ipycache works right now
RUN pip install git+https://github.com/rossant/ipycache.git

# Install the baselines repo, which will show up as the ocpmodels python module
RUN pip install git+https://github.com/Open-Catalyst-Project/baselines.git

# Expose the `/home/.config` folder so that it can be mounted when the
# container is opened. This will enable us to use wandb.
VOLUME $HOME/.config

# Make the default user `user` instead of `root`. Necessary when working with Shifter.
RUN chown -R user:group $HOME
USER user
