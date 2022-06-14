# NLE Toolbox

A collection of tools, wrappers, observation preprocessors, and other useful
tools for the [Nethack Learning Environment](https://github.com/facebookresearch/nle.git).

## Installation

The following command sets up an env with the essential dependencies required by the toolbox: `torch`, `numpy`, [`nle`](https://github.com/facebookresearch/nle.git), [`python-plyr`](https://github.com/ivannz/plyr.git), [minihack](https://github.com/facebookresearch/minihack.git), and others.

```bash
# conda deactivate && conda env remove -n nle_toolbox

conda create -n nle_toolbox "python>=3.9" pip setuptools numpy scipy \
  "pytorch::pytorch>=1.8" matplotlib \
  && conda activate nle_toolbox \
  && pip install einops

# install bleeding-edge versions from the git
# NLE <= 0.8.1 does not expose certain bottom line attribs
pip install -vv "nle @ git+https://github.com/facebookresearch/nle.git"

# this fork of minihack patched missing parameters in some tasks
pip install -vv "minihack @ git+https://github.com/ivannz/minihack.git"

# this branch features optimization that have not been merged yet
pip install -vv "python-plyr @ git+https://github.com/ivannz/plyr.git@populate"
# TODO replace with `pip install python-plyr` when available on pypi
```

The latest version of the toolbox itself can be installed from github

```bash
conda activate nle_toolbox

pip install git+https://github.com/ivannz/nle_toolbox.git -vv
```

Or if you would like to contribute or edit/fix some code or tools contained within, please, clone and install the package in editable mode. It is advisable to create a dedicated environments for development purposes.

```bash
git clone https://github.com/ivannz/nle_toolbox.git
cd nle_toolbox

# use a dedicated env with correct dependencies
conda activate nle_toolbox

conda install -n nle_toolbox swig pytest notebook \
  && pip install black pre-commit gitpython box2d-py "gym[box2d]"
# XXX gitpython and gym[box2d] for fancy testing

pre-commit install

pip install -e . -vv
```
