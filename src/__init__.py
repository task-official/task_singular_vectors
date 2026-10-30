"""Initialize NumPy/MKL before PyTorch loads its OpenMP runtime.

This also makes CLI subprocesses launched from NumPy-based notebooks behave
consistently, without overriding the user's MKL threading environment.
"""
import numpy
