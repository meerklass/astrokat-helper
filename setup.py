from setuptools import setup, find_packages

setup(
    name='astrokat-helper',
    version='0.0.1',
    description='Helper scripts for astrokat',
    author='',
    author_email='',
    packages=find_packages(),
    install_requires=[
        'astrokat',
    ],
    entry_points={
        'console_scripts': [
            'when_is_patch_observable = scripts.when_is_patch_observable:main',
        ]
    },
    classifiers=[
        'GPLv3',
        'Programming Language :: Python :: 3.10',
    ],
)
