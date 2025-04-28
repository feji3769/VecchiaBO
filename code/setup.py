from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
import glob


root_dir = os.path.dirname(os.path.abspath(__file__))
sorting_dir = os.path.join(root_dir, 'pyvecch', 'sorting')
source_cpu = glob.glob(os.path.join(sorting_dir, '*.cpp'))

ext_modules = [
    CppExtension(
        name='_pyvecch',
        sources=source_cpu,
        include_dirs=[sorting_dir],
    )
]

setup(
    name='pyvecch',
    version='0.0.1',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
