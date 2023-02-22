from setuptools import setup, find_packages

setup(
    name="pyvecch",
    version="0.0.1",
    packages=find_packages()
)


from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
import glob

include_dirs = os.path.dirname(os.path.abspath(__file__)) 
include_dirs = os.path.join(include_dirs, 'pyvecch', 'sorting')
source_cpu = glob.glob(os.path.join(include_dirs, '*.cpp'))

ext_module = [cpp_extension.CppExtension(
  '_pyvecch', 
  source_cpu,
  include_dirs = [include_dirs]
  )]

setup(name='pyvecch',
      ext_modules=ext_module,
      cmdclass={'build_ext': cpp_extension.BuildExtension}, 
      packages=find_packages())
