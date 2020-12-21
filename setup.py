import os
import re
import sys

import numpy as np
from setuptools import setup
from setuptools.extension import Extension

# determine version
with open('mvptree/__init__.py', 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

path = os.path.abspath(os.path.dirname(__file__))
ext_kwargs = {
    'language': 'c++',
    'include_dirs': [np.get_include()],
    'extra_compile_args': ['-std=c++11']
}

# run cython if specified
if len(sys.argv) >= 2 and sys.argv[1].lower() == 'cython':
    from Cython.Build import cythonize

    cythonize([Extension('mvptree.core', sources=[os.path.join('mvptree', 'core.pyx')], **ext_kwargs)], 
              compiler_directives={'language_level': 3, 
                                   'boundscheck': False, 
                                   'wraparound': False,
                                   'cdivision': True},
              annotate=True)

else:
    extensions = [Extension('mvptree.core', sources=[os.path.join('mvptree', 'core.cpp')], **ext_kwargs)]

    # other options specified in setup.cfg
    setup(version=__version__, ext_modules=extensions)
