from setuptools import setup, find_packages

setup(
    name = "INTI", # Replace with your own username
    version = "1.0.0",
    description = "A python package to calculate molecular cross sections for substellar atmospheres.",
    packages = ['INTI'],
    include_package_data = False,
    python_requires = '>=3.7.6',
    install_requires = ['numpy',
                        'scipy',
                        'matplotlib',
                        'h5py',
                        'numba>=0.56',
                        'requests',
                        'bs4',
                        'tqdm',
                        'pandas',
                        'lxml',
                        'hitran-api',
                        'pytest',
                        'jupyter'],
    zip_safe = False,
)