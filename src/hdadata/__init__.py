from .load import Loader
from .structure import HDAData
from . import visualize


def load_scifate():
    loader = Loader()
    return loader.load_as_hda()

__all__ = [
    "Loader",
    "HDAData"
]