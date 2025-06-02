from setuptools import setup, find_packages

# Read the long description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lab_tools",
    version="0.2.3",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.0",  # Numerical computations
        "scipy>=1.11.0",  # Scientific computing
        "sympy>=1.12",  # Symbolic mathematics
        "pandas>=2.0.0",  # Data manipulation and analysis
    ],
    description="A custom library for report calculations, including data analysis and uncertainty measurement tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="jakub627",
    author_email="dybich89@gmail.com",
    url="https://github.com/jakub627/lab_tools",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Tracker": "https://github.com/jakub627/lab_tools/issues",
        "Documentation": "https://github.com/jakub627/lab_tools#readme",
        "Source Code": "https://github.com/jakub627/lab_tools",
    },
)
