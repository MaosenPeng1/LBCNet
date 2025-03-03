# LBCNet: Local Balance and Calibration-Based Propensity Score Estimation

## Description

LBCNet implements LBC-Net for estimating propensity scores, as introduced in "A Deep Learning Approach to Nonparametric Propensity 
Score Estimation with Optimized Covariate Balance." This package proposes a novel propensity score weighting method based on two fundamental 
conditions: local balance, which ensures conditional independence of covariates and treatment assignment across a dense grid of balancing scores, 
and local calibration, which guarantees that the balancing scores are mapped to the true propensity scores. By leveraging a neural network, LBCNet 
develops a nonparametric propensity score model that effectively optimizes covariate balance, minimizes bias, and stabilizes inverse probability of 
treatment weights (IPTW).

## Installing Python

LBCNet requires Python to be installed on your system. You can download and install Python from the official website:

- [Download Python](https://www.python.org/downloads/)

Make sure to install **Python 3.8 - 3.11**, as versions **3.12 and above** may cause compatibility issues with some dependencies.

### Verifying Python Installation

After installing Python, open a terminal or command prompt and check if Python is installed correctly by running:

```sh
python --version
```
or

```sh
python3 --version
```
If Python is installed correctly, you should see an output similar to:

```sh
Python 3.10.12
```

For Windows users, ensure that Python is added to the system PATH during installation. If you encounter any issues, 
refer to the [official Python documentation](https://docs.python.org/3/using/windows.html) for troubleshooting.

## Installing and Configuring Reticulate in R

LBCNet uses the `reticulate` package to interface between R and Python. To install `reticulate`, run the following command in R:

```r
install.packages("reticulate")
```
For detailed installation and setup instructions, visit the [Reticulate Documentation](https://rstudio.github.io/reticulate/).
After installing `reticulate`, load the package and check the current Python configuration:

```r
library(reticulate)

# Display the Python version and path being used
py_config()
```
If the detected Python version is not the desired one, you can manually set the Python path. For example:

```r
use_python("C:/Users/YourUsername/AppData/Local/Programs/Python/Python311/python.exe", required = TRUE)
```
**Note**: Ensure that the specified path matches your Python installation directory.

### Options for Setting Up Python

There are multiple ways to configure Python for `reticulate`:

1. System Python (Default)
- Uses the Python installation detected on the system.
- Can be set manually with use_python("path/to/python").
- Works well if dependencies are already installed globally.

2. Virtual Environment (`venv`)
- Creates an isolated environment for Python dependencies.
- Set up with:
```r
virtualenv_create("r-reticulate")
use_virtualenv("r-reticulate")
```
- Helps avoid conflicts with system Python and ensures reproducibility.

3. Conda Environment
- Uses a Conda-managed Python environment.
- Set up with:
```r
conda_create("r-lbcnet", packages = c("python=3.11"))
use_condaenv("r-lbcnet")
```
- Useful for handling package dependencies but requires Conda installation.

### Common Issues and Fixes

1. **Multiple Python Installations**: If you have multiple versions of Python installed, you may need to specify the correct path using `use_python()`.
2. **Administrator Privileges**: Some installations require running R with administrator privileges to install dependencies.
3. **Dependency Restrictions**: Certain Python packages may not work with the latest versions of Python (e.g., Python ≥3.12).

## Installing Required Python Packages

After setting up Python with `reticulate`, you need to install the necessary Python dependencies for LBCNet. The required packages include:
- `torch` 
- `numpy` 
- `pandas` 
- `tqdm` 

### Installing Python Packages

You can install the required Python packages using either the system’s command line or directly from R:

#### Option 1: Using System Commands

Run the following command in R or in command prompt to install the packages via `pip`:

```r
system("pip install torch numpy pandas tqdm")
```
```sh
pip install torch numpy pandas tqdm
```
#### Option 2: Using reticulate in R
Refer to [Python Package Instruction](https://rstudio.github.io/reticulate/articles/python_packages.html)
for detailed guide. For local intallation, install the packages directly from R using `py_install()`:
```py
py_install(c("numpy", "pandas", "torch", "tqdm"))
```
### Verifying Installation
After installing the required packages, verify that they were installed correctly by running the following commands:

```r
library(reticulate)

# Check if Python is correctly configured
py_config()

# Run Python commands from R
py_run_string("print('Hello from Python!')")  # Should print "Hello from Python!"

# Check package versions
py_run_string("import numpy; print(numpy.__version__)")
py_run_string("import torch; print(torch.__version__)")
```
If all commands execute without errors and display package versions, the installation was successful.

## Install LBCNet

```r
devtools::install_github("MaosenPeng1/LBCNet")
```
Or

```r
remotes::install_github("MaosenPeng1/LBCNet")
```
