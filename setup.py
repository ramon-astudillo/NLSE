import os
from setuptools import setup, find_packages

try:
    from pip.req import parse_requirements
    import pip.download

    # parse_requirements() returns generator of pip.req.InstallRequirement
    # objects
    install_reqs = parse_requirements(
        "requirements.txt",
        session=pip.download.PipSession()
    )
    # install_requires is a list of requirement
    install_requires = [str(ir.req) for ir in install_reqs]
except:
    # This is a bit of an ugly hack, but pip is not installed on EMR
    install_requires = []


# Utility function to read the README file.
def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


package_data = {}

setup(
    name='NLSE',
    version='0.0.2',
    author='@ramon-astudillo',
    author_email='ramon@astudillo.com',
    description="Code for the NLSE model in SemEval 2015, 2016.",
    long_description=read('README.md'),
    license='MIT',
    url='https://github.com/ramon-astudillo/NLSE',
    py_modules=['nlse'],
    packages=find_packages(),
    install_requires=install_requires,
    package_data=package_data,
)
