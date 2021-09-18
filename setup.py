import os
from setuptools import setup


if __name__ == '__main__':
    # update the version number
    version = open('VERSION', 'r').read().strip()

    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cwd, 'nle_tools', '__version__.py'), 'w') as f:
        f.write(f'__version__ = \'{version}\'\n')

    setup(
        name='nle_tools',
        version=version,
        description="""Toolbox for NLE Challenge""",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        license='MIT',
        packages=[
            'nle_tools',
        ],
    )
