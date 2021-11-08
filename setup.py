import pathlib

from setuptools import find_packages, setup

import tom_rapperson

_THIS_DIR = pathlib.Path(__file__).parent


def _get_requirements():
    with (_THIS_DIR / 'requirements.txt').open() as fp:
        return fp.read()


setup(
    name='tom_rapperson',
    version=tom_rapperson.__version__,
    install_requires=_get_requirements(),
    package_dir={'tom_rapperson': 'tom_rapperson'},
    packages=find_packages(exclude=['tests', 'tests.*']),
)
