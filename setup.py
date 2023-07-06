# from distutils.core import setup
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([
        Extension(name='geophy.tree_dec.nj', sources=['geophy/tree_dec/nj.pyx'], include_dirs=[numpy.get_include()]),
    ],
    annotate=True, language_level="3"),
)
