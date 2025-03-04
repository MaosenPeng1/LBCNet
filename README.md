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

## First-Time Setup: Installing and Configuring Reticulate in R

LBCNet uses the reticulate package to interface between R and Python.
The first time you use LBCNet, you must set up reticulate and configure Python.
Once the setup is complete, you won’t need to configure it every time.

### 1.  Install `reticulate` in R

If you haven’t installed reticulate yet, run:
```r
install.packages("reticulate")
```
For detailed installation and setup instructions, visit the [Reticulate Documentation](https://rstudio.github.io/reticulate/).

### 2.  Verify Python Installation in R

After installing `reticulate`, load it and check which Python version is detected:

```r
library(reticulate)

# Display the Python version and path being used
py_config()
```
If `reticulate` detects the correct Python version, you can skip the next step.
If not, you must manually specify the correct Python path.

### 3. First-Time Python Setup: Choose One of the Following Options
There are multiple ways to configure Python for `reticulate`.
Choose one method that best suits your setup.

Option 1: Use System Python (Default)
If Python is installed globally on your system, `reticulate` should detect it automatically.
To manually specify the path:
```r
use_python("C:/Users/YourUsername/AppData/Local/Programs/Python/Python311/python.exe", required = TRUE)
```
Best for: Users with Python already installed globally and dependencies manually managed.
Using system Python may cause conflicts if other R packages require different dependencies.

Option 2: Create a Virtual Environment (Recommended)
A virtual environment (venv) isolates Python dependencies, ensuring LBCNet runs without conflicts.
Create and activate a virtual environment:
```r
virtualenv_create("r-lbcnet")
use_virtualenv("r-lbcnet", required = TRUE)
```
Best for: Ensuring package isolation and avoiding conflicts with other Python versions.

Option 3: Use a Conda Environment
If you have Conda installed, you can use a Conda-managed Python environment.
```r
conda_create("r-lbcnet", packages = c("python=3.11"))
use_condaenv("r-lbcnet", required = TRUE)
```
Best for: Users who already use Conda to manage Python dependencies.

### 4. First-Time Installation of Required Python Packages
Once Python is configured, you need to install the required dependencies.
Run one of the following:
```r
py_install(c("torch", "numpy", "pandas", "tqdm"))
```
OR
```r
system("pip install torch numpy pandas tqdm")
```

Verify the installation:
```r
py_run_string("import numpy; print(numpy.__version__)")
py_run_string("import torch; print(torch.__version__)")
```
For detailed package intall instructions, visit the [Package Install](https://rstudio.github.io/reticulate/articles/python_packages.html).

### 5. Common Issues and Fixes

1. **Multiple Python Installations**: If you have multiple versions of Python installed, you may need to specify the correct path using `use_python()`.
2. **Administrator Privileges**: Some installations require running R with administrator privileges to install dependencies.
3. **Dependency Restrictions**: Certain Python packages may not work with the latest versions of Python (e.g., Python ≥3.12).
4. **Python version mismatch**: Run `py_config()` and ensure Python is set to the correct version.
5. **Module not found (e.g., torch not found)**: Run `py_install("torch")` to install missing dependencies or try `py_require("torch")`.
6. **Failed to initialize Python**: Restart your R session (`Session` > `Restart R`) and rerun `use_virtualenv()` or `use_condaenv()`.


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
