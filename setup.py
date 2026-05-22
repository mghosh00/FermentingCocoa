from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the fermenting_cocoa module.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('fermenting_cocoa'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_requirements():
    requirements = []
    with open("requirements.txt", "r") as file:
        for line in file:
            requirements.append(line)
    return requirements


setup(
    # Module name
    name='fermenting_cocoa',

    # Version
    version=get_version(),

    description='Well-mixed model of the fermentation of cocoa pulp',

    maintainer='Matthew Ghosh, Mona Li',

    maintainer_email='matthew.ghosh@gtc.ox.ac.uk',

    url='https://github.com/mghosh00/FermentingCocoa',

    # Packages to include
    packages=find_packages(include=('fermenting_cocoa', 'fermenting_cocoa.*')),

    # List of dependencies
    install_requires=get_requirements(),

    extras_require={
        'docs': [
            'sphinx>=1.5, !=1.7.3',
        ],
        'dev': [
            'flake8>=3',
            'pytest',
            'pytest-cov',
        ],
    },
)
