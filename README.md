## Introduction

The `ebsl` is a Python module implementing the Ensemble Binomial Subjective Logic framework, a binary ensemble classifier (based on subjective logic) that provides dynamic weighting.

The C++ extension is documented [here](https://a-h-ismail.gitlab.io/ebsl-ids-docs/).

## Installation

The EBSL framework is installable using `pip`. Simply run `pip install ebsl`. The instructions below are for building from source.

### Dependencies

- Python (>= 3.12)
- Numpy (>= 2.0)
- Pandas

You will also need a C++20 compliant compiler to compile the C++ extension and `cmake`.

### Building using pip

Clone the repository and move into its directory:
```
git clone --recurse-submodules https://github.com/a-h-ismail/ebsl-ids.git
cd ebsl-ids
```

Then install using `pip`:
```
pip install .
```
The extension will be automatically built and installed globally. Adapt the command as needed for virtual environments.
