# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:52:07 2019

@author: ssj
"""

"""
The build/compilations setup
>> pip install -r requirements.txt
>> python setup.py install
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

#
#def _parse_requirements(file_path):
#    pip_ver = pkg_resources.get_distribution('pip').version
#    pip_version = list(map(int, pip_ver.split('.')[:2]))
#    if pip_version >= [6, 0]:
#        raw = pip.req.parse_requirements(file_path,
#                                         session=pip.download.PipSession())
#    else:
#        raw = pip.req.parse_requirements(file_path)
#    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
#try:
#    install_reqs = _parse_requirements("requirements.txt")
#except Exception:
#    logging.warning('Fail load requirements file, so using default ones.')
#    install_reqs = []

setup(
    name='siim',
    version='1',
    url='https://github.com/dhirensk/siim',
    author='dhirensk',
    author_email='dhirensk@gmail.com',
    license='None',
    description='Mask R-CNN for object detection and instance segmentation',
    packages=["mrcnn"],
    #install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.4',
    long_description="""This is test. """,
    classifiers=[
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
    #keywords="image instance segmentation object detection mask rcnn r-cnn tensorflow keras",
)