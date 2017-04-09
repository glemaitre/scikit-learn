import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("regression_tree", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension("criterion",
                         sources=["criterion.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("stats_node",
                         sources=["stats_node.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("split_record",
                         sources=["split_record.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("splitter",
                         sources=["splitter.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_subpackage("tests")

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
