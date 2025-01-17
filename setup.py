from setuptools import setup, find_packages

setup(
    name="lab_tools",
    version="0.2.2",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.0",  # Zmieniona minimalna wersja
        "scipy>=1.11.0",  # Zmieniona minimalna wersja
    ],
    description="A custom library for report calculations, including data analysis and uncertainty measurement tools.",
    author="jakub627",
    author_email="dybich89@gmail.com",
    url="https://github.com/yourusername/lab_tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
)
