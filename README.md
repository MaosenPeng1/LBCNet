# LBCNet: Local Balance and Calibration-Based Propensity Score Estimation

## Description

LBCNet implements LBC-Net for estimating propensity scores, as introduced in "A Deep Learning Approach to Nonparametric Propensity 
Score Estimation with Optimized Covariate Balance." This package proposes a novel propensity score weighting method based on two fundamental 
conditions: local balance, which ensures conditional independence of covariates and treatment assignment across a dense grid of balancing scores, 
and local calibration, which guarantees that the balancing scores are mapped to the true propensity scores. By leveraging a neural network, LBCNet 
develops a nonparametric propensity score model that effectively optimizes covariate balance, minimizes bias, and stabilizes inverse probability of 
treatment weights (IPTW).

In addition to estimating ATE and ATT for binary treatments, LBCNet provides a unified interface for:

- **Variance estimation**: plug-in influence functions, model-based uncertainty quantification, and 
  robust standard errors for IPTW and LBC-Net–based estimators.
- **Survival estimands**: risk difference and survival difference at a fixed time \( t \), with 
  support for IPW–adjusted Kaplan–Meier and Nelson–Aalen estimators.

The package integrates R and Python modules to deliver a scalable, easy-to-use workflow for modern 
causal inference research, including treatment effect estimation, sensitivity analysis, and evaluation of 
covariate balance under learned propensity score models.

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
available_pythons <- unique(c(
  Sys.which("python"),
  Sys.which("python3"),
  reticulate::py_discover_config()$python
))

available_pythons <- available_pythons[nzchar(available_pythons)]  # Remove empty results
print(available_pythons)

# Check available virtual and Conda environments
reticulate::virtualenv_list()
reticulate::conda_list()
```
If `reticulate` detects the correct Python version, you can skip the next step.
If not, you must manually specify the correct Python path.

### 3. First-Time Python Setup: Choose One of the Following Options
There are multiple ways to configure Python for `reticulate`.
Choose one method that best suits your setup.

**Option 1**: Use System Python (Recommended for Servers & Clusters)
If Python is installed globally on your system, `reticulate` should detect it automatically.
To manually specify the path:
```r
## Set the Python Path for the Entire R Session 
## Best for: Servers, clusters, and users who manage Python separately.
Sys.setenv(RETICULATE_PYTHON = "/path/to/python")  # Adjust based on your system
reticulate::py_discover_config()
```
Or

```r
## Set Python for the Current R Session Only
## Best for: Local machines where Python paths may change frequently.
reticulate::use_python("/path/to/python", required = TRUE)
reticulate::py_config()
```
This ensures stability by using a known, system-managed Python installation. 
Using system Python may cause conflicts if other R packages require different dependencies.

**Option 2**: Create a Virtual Environment (Recommended)
A virtual environment (venv) isolates Python dependencies, ensuring LBCNet runs without conflicts.
Create and activate a virtual environment:
```r
reticulate::virtualenv_create("r-lbcnet")  # Create virtual environment
reticulate::use_virtualenv("r-lbcnet", required = TRUE)  # Activate virtual environment
```
Best for: Ensuring package isolation and avoiding conflicts with other Python versions.

**Option 3**: Use a Conda Environment
If you have Conda installed, you can use a Conda-managed Python environment.
```r
reticulate::conda_create("r-lbcnet", packages = c("python=3.11"))
reticulate::use_condaenv("r-lbcnet", required = TRUE)
```
Best for: Users who already use Conda to manage Python dependencies.

### 4. First-Time Installation of Required Python Packages
Once Python is configured, you need to install the required dependencies.
Run one of the following:
```r
reticulate::py_install(c("torch", "numpy", "pandas", "tqdm"), envname = "r-lbcnet")
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

If all commands execute without errors and display package versions, the installation was successful.

## Install LBCNet

```r
devtools::install_github("MaosenPeng1/LBCNet")
```
Or

```r
remotes::install_github("MaosenPeng1/LBCNet")
```
