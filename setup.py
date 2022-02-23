from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='My numerics',
    url='https://github.com/nemectad/mynumerics',
    author='Jan Vabek, Tadeas Nemec',
    author_email='nemectad@fjfi.cvut.cz',
    # Needed to actually package something
    packages=['mynumerics'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'h5py'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Module containing numerical methods and utilities for HHG simulations.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)