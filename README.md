# NLE Toolbox

A collection of tools, wrappers, observation preprocessors, and other useful
tools for the [Nethack Learning Environment](https://github.com/facebookresearch/nle.git).

## Installation

The following command sets up an env with the essential dependencies required by the toolbox: `torch`, `numpy`, [`nle`](https://github.com/facebookresearch/nle.git), [`python-plyr`](https://github.com/ivannz/plyr.git), and others.

```bash
conda create -n nle_toolbox "python>=3.9" pip setuptools numpy scipy \
  "pytorch::pytorch>=1.8" matplotlib \

conda activate nle_toolbox \
  && pip install "python-plyr>=0.8" einops nle minihack
```

The latest version of the toolbox itself can be installed from github

```bash
conda activate nle_toolbox

pip install git+https://github.com/ivannz/nle_toolbox.git
```

Or if you would like to contribute or edit/fix some code or tools contained within, please, clone and install the package in editable mode. It is advisable to create a dedicated environments for development purposes.

```bash
git clone https://github.com/ivannz/nle_toolbox.git
cd nle_toolbox

# use a dedicated env with correct dependencies
conda activate nle_toolbox

conda install -n nle_toolbox swig pytest black pre-commit \
  && pip install gitpython box2d-py "gym[box2d]<=0.23"
# XXX gitpython and gym[box2d] for fancy testing

pre-commit install

pip install -e .
```
