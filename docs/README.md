# EBSL-IDS C++ Extension Module

## Introduction

This documentation is meant only for the C++ extension of the Ensemble Binomial Subjective Logic Framework. The Python wrapper comes with its own docstrings.

## Building from Source

Refer to the main project's README file.

## Usage

The C++ extension implements three classes: Opinion, BSL_SM and EBSL. They are exposed to Python using `nanobind` under the names `Opinion`, `BSL_SM_cpp` and `EBSL_cpp`. To keep the implementation simple, the C++ extension is only concerned with running the algorithm on data it receives from Python as numpy arrays. You can check the classes documentation to learn more and the `ebsl_pywrapper.py` file to see an example wrapping `BSL_SM_cpp` and `EBSL_cpp` to add support for scikit-learn models.
