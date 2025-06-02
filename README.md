# Lab Tools Library

`lab_tools` is a Python library designed to assist with various laboratory-related calculations and data analysis. It provides tools for uncertainty calculations, linear regression, data plotting, and rounding operations. It is built to support scientific projects that require handling datasets, performing statistical analysis, and visualizing results.

## Features

- **Uncertainty Calculations**: Functions for calculating uncertainties in measurements.
- **Linear Regression**: Classes and functions for performing linear regression on datasets.
- **Data Plotting**: Tools for visualizing data using various plots.
- **Rounding**: Functions to round results to the desired precision.
- **Mean Uncertainty**: Functions for calculating mean uncertainties across datasets.

## Installation

To install `lab_tools`, you can use pip. If you’re installing from the GitHub repository, use the following command:

```bash
pip install git+https://github.com/jakub627/lab_tools.git
```

Alternatively, clone the repository and install it manually:

```bash
git clone https://github.com/jakub627/lab_tools.git
cd lab_tools
pip install .
```

## Usage

### Example of Using `linear_regression`

```python
from lab_tools.linear_regression import LinearRegression

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a LinearRegression object
lr = LinearRegression(x, y)

# Perform linear regression
slope, intercept, stderr, intercept_stderr, rvalue = lr.fit()

# Print results
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"Stderr: {stderr}")
print(f"Intercept Stderr: {intercept_stderr}")
print(f"R Value: {rvalue}")


```

### Example of Using `mean_uncertainty`

```python
from lab_tools.mean_uncertainty import MeanUncertainty

# Example uncertainties
values = [0.1, 0.2, 0.15, 0.25]

# Calculate mean uncertainty
mean, stderr = MeanUncertainty(values)
print(f"Mean: {mean}")
print(f"Stderr: {stderr}")

```

### Directory Structure

```plaintext
lab_tools/
├── __init__.py
├── checking.py
├── linear_regression.py
├── mean_uncertainty.py
├── plotting.py
└── rounding.py
```

- `checking.py`: Functions for data validation and error checking.
- `linear_regression.py`: Contains the LinearRegression class and methods for linear regression.
- `mean_uncertainty.py`: Contains functions for calculating the mean uncertainty of data.
- `plotting.py`: Includes functions for generating data visualizations.
- `rounding.py`: Functions for rounding numbers to specified precision.

## Requirements

- Python 3.x
- numpy
- scipy
- pandas

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on the repository.
