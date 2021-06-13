from setuptools import setup, find_packages

VERSION = '0.0.5'
DESCRIPTION = 'Client library to create serverless REST endpoints on shippedbrain.com'
LONG_DESCRIPTION = open("README.md", "r", encoding="utf-8").read()

install_requires = [
      "mlflow",
      "click"
]

extras_require = {
      "dev": ["pytest"]
}

# Setting up
setup(name="shippedbrain",
      version=VERSION,
      author="Shipped Brain",
      author_email="info@shippedbrain.com",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      python_requires=">=3.6.0",
      install_requires=install_requires,
      extras_require=extras_require,
      keywords=['pretrained-models', 'shipped-brain', 'machine-learning', 'artificial-intelligence', 'deploy', 'serve', "serverless", "model-hub", "data-science", 'mlflow'],
      license="LICENSE",
      url="https://github.com/shippedbrain/shippedbrain",
      entry_points={
            "console_scripts": [
                  "shippedbrain=shippedbrain.cli:cli"
            ]
      },
      classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ])
