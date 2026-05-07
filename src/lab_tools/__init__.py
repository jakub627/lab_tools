from .config import Formatters, Locators, AxesUtils, Norms, SI, LoggerFactory
from .curve_fit import CurveFit
from .file_io import PICKLE, JSON, TXT
from .linear_regression import LinearRegression
from . import wave
from . import typing

__all__ = [
    "Formatters",
    "Locators",
    "AxesUtils",
    "Norms",
    "SI",
    "LoggerFactory",
    "CurveFit",
    "PICKLE",
    "JSON",
    "TXT",
    "LinearRegression",
    "wave",
    "typing",
]

__version__ = "1.3.0a10"
