import os
from setuptools import setup


if __name__ == "__main__":
    # update the version number
    version = open("VERSION", "r").read().strip()

    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cwd, "nle_toolbox", "__version__.py"), "w") as f:
        f.write(f"__version__ = '{version}'\n")

    setup(
        name="nle_toolbox",
        version=version,
        description="""Toolbox for NLE Challenge""",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        license="MIT",
        packages=[
            "nle_toolbox",
            "nle_toolbox.utils",
            "nle_toolbox.utils.env",
            "nle_toolbox.utils.env.minihack",
            "nle_toolbox.utils.replay",
            "nle_toolbox.utils.rl",
            "nle_toolbox.bot",
            "nle_toolbox.bot.model",
            "nle_toolbox.zoo",
            "nle_toolbox.zoo.models",
            "nle_toolbox.zoo.transformer",
            "nle_toolbox.zoo.vq",
        ],
        python_requires=">=3.9",
        install_requires=[
            "numpy",
            "torch>=1.8",
            "python-plyr",
            "scipy",
            "einops",
            "gym<=0.23",
            "nle>=0.8.0",
            "minihack",
            "matplotlib",
        ],
        test_requires=[
            "gitpython",
            "pytest",
            "swig",
            "box2d-py",
            "gym[box2d]<=0.23",
        ],
    )
