from setuptools import setup, find_packages

setup(
    name="recruitment-candidate-selection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.109.2",
        "uvicorn==0.27.1",
        "pydantic==2.6.1",
        "numpy==1.26.3",
        "pandas==2.2.0",
        "scikit-learn==1.4.0",
        "matplotlib==3.8.2",
        "seaborn==0.13.2",
        "faker==22.6.0",
        "python-dotenv==1.0.0",
        "pytest==8.0.0"
    ],
) 