

--------------------------------------------------------------------------------
This branch of Asteroid contains code for the ensemble separation related papers below:
* [x] [Choral Music Separation using Time-domain source separation] (./egs/MedleyDB/ConvTasNet) ([Sarkar et al.](https://www.isca-archive.org/interspeech_2021/sarkar21_interspeech.pdf))
* [x] [EnsembleSet: A new high-quality synthesised dataset for chamber ensemble separation] (./asteroid/data/bbcso_dataset.py) ([Sarkar et al.](https://archives.ismir.net/ismir2022/paper/000075.pdf))
* [x] [Leveraging Synthetic Data for Improving Chamber Ensemble Separation] (/egs/BBCSO/DPTNet) ([Sarkar et al.](https://ieeexplore.ieee.org/iel7/10248019/10248047/10248118.pdf?casa_token=CM4xFMKQuYUAAAAA:cQ-j2wzUz1Ac_07Ml2BcAy3HZyNau7wOjuI0RqvIw2ph9GBHY-obArJOiOlCeeKGHuy_t5qY))

This repo is forked from Asteroid, which is a Pytorch-based audio source separation toolkit
that enables fast experimentation on common datasets.
It comes with a source code that supports a large range
of datasets and architectures, and a set of
 recipes to reproduce some important papers. 


## Contents
- [Installation](#installation)
- [Running a recipe](#running-a-recipe)
- [Available recipes](#available-recipes)
- [Supported datasets](#supported-datasets)
- [Citing us](#citing)

## Installation
([↑up to contents](#contents))
To install Asteroid, clone the repo and install it using
conda, pip or python :
```bash
# First clone and enter the repo
git clone https://github.com/saurjya/EnsembleSep
cd EnsembleSep
```

- With `pip`
```bash
# Install with pip in editable mode
pip install -e .
# Or, install with python in dev mode
# python setup.py develop
```
- With conda (if you don't already have conda, see [here][miniconda].)
```bash
conda env create -f environment.yml
conda activate asteroid
```

- Asteroid is also on PyPI, you can install the latest release with
```bash
pip install asteroid
```


## Running a recipe
([↑up to contents](#contents))
Running the recipes requires additional packages in most cases,
we recommend running :
```bash
# from asteroid/
pip install -r requirements.txt
```
Then choose the recipe you want to run and run it!
```bash
cd egs/BBCSO/DPTNet
. ./run.sh
```
More information in [egs/README.md](./egs).

## Available recipes
([↑up to contents](#contents))
* [x] [ConvTasnet](./egs/wham/ConvTasNet) ([Luo et al.](https://arxiv.org/abs/1809.07454))
* [x] [DualPathRNN](./egs/wham/DPRNN) ([Luo et al.](https://arxiv.org/abs/1910.06379))
* [x] [DPTNet](./asteroid/models/dptnet.py) ([Chen et al.](https://arxiv.org/abs/2007.13975))
* [x] [DCUNet](./asteroid/models/dcunet.py) ([Choi et al.](https://arxiv.org/abs/1903.03107))

## Supported datasets
([↑up to contents](#contents))
* [x] [EnsembleSet](./egs/BBCSO) / BBCSO ([Sarkar et al.](https://archives.ismir.net/ismir2022/paper/000075.pdf))

