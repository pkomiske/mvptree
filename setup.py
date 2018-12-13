from setuptools import setup, find_packages
from setuptools.extension import Extension

import os
import platform
import re
import sys

import numpy as np

# determine version
with open('mvptree/__init__.py', 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                            f.read()).group(1)

# determine if we use cython or not
cythonize, ext, extpp = False, '.c', '.cpp'
if len(sys.argv) >= 2 and sys.argv[1] == 'build_ext':
    from Cython.Build import cythonize
    ext = extpp = '.pyx'

opt_args = ['-O3']
if platform.system() == 'Darwin' and platform.release() == '18.0.0':
    opt_args.append('-stdlib=libc++')

path = os.path.abspath(os.path.dirname(__file__))
extensions = [
    Extension(
        'mvptree.lp.emd_wrap',
        sources=['mvptree/lp/emd_wrap'+extpp, 'mvptree/lp/EMD_wrapper.cpp'],
        language='c++',
        include_dirs=[np.get_include(), os.path.join(path, 'mvptree/lp')],
        extra_compile_args=opt_args
    ),
    Extension(
        'mvptree.mvptree',
        sources=['mvptree/mvptree'+ext],
        include_dirs=[np.get_include()],
        extra_compile_args=opt_args
    ),
]

if cythonize:
    extensions = cythonize(extensions, 
                          compiler_directives={'language_level': 3, 
                                               'boundscheck': False, 
                                               'wraparound': False,
                                               'cdivision': True},
                          annotate=True)

# other options specified in setup.cfg
setup(
    version=__version__,
    ext_modules=extensions
)