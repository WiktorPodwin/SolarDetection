from setuptools import setup, find_packages

NAME = "solar_detection"
VERSION = "0.1"
DESCRIPTION = "Solar Detection"
LONG_DESCRIPTION = "Solar Detection"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(exclude=("data",)),
    package_dir={"solar_detection": "src"},
    install_requires=[
        "numpy<2",
        "opencv-python==4.10.0.84",
        "selenium==4.25.0",
        "webdriver-manager==4.0.2",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "timm==1.0.11",
        "pillow-heif==0.20.0",
        "matplotlib==3.9.2",
        "pandas==2.2.3",
        "seaborn==0.13.2",
        "scikit-learn==1.5.2",
    ],
)
