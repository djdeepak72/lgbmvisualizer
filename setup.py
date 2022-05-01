from setuptools import setup

exec(open("lgbmvisualizer/version.py").read())

install_requires = [
    "numpy",
    "pandas",
    "lightgbm==3.2.1",
    "shap==0.39.0",
    "seaborn",
    "matplotlib",
]

setup(
    name="lgbmvisualizer",
    description="A python package that uses LGBM model to find & plot interactions between your variables in your model",
    author="DJ",
    author_email="willofdeepak@gmail.com",
    install_requires=install_requires,
    license="",
    packages=["lgbmvisualizer"],
    python_requires=">=3.7",
    version=__version__
)