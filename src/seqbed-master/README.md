# seqbed
A Python package to perform sequential Bayesian experimental design for implicit models via mutual information, as in https://arxiv.org/abs/2003.09379.

## Installation
Create and activate a virtual environment, then pip install the package.
For example, with conda:

```
conda create -n env python=3.6
conda activate env
```

Then to install the package, change directory to the root of the package and:
```
pip install .
conda install scikit-learn=0.21
pip install --upgrade scipy==1.4.1 --user
```
