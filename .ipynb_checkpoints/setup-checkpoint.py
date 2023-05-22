from setuptools import setup

setup(
    name="HousingPradeep",
    version="1.0.0",
    description="A package to download, train and scoring models using housing dataset",
    author="Sai Kapu",
    author_email="sai.kapu@tigeranalytics.com",
    packages=["src", "src/mle_lib"],  # same as name
    install_requires=[],  # external packages as dependencies
    scripts=[],
)

