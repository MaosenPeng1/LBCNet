#' Setup Python Environment for LBCNet
#'
#' @description This function configures the Python environment for LBCNet, ensuring the correct
#' Python version and dependencies are loaded. It prioritizes virtual environments (`venv`) over
#' Conda unless explicitly requested. Additionally, users can explicitly choose to use the system Python.
#'
#' @param use_conda Logical. If `TRUE`, it attempts to use Conda (`r-lbcnet`) instead of virtualenv.
#'   Default is `FALSE`, meaning virtualenv/system Python is preferred.
#' @param envname A character string specifying the name of the virtual environment or Conda environment
#'   to use. Default is `"r-lbcnet"`. One can use `virtualenv_list()` or `conda_list()` to check the available Python environments in your system.
#' @param use_system_python Logical. If `TRUE`, the function will force the use of the system Python (`Sys.which("python")`) instead of a virtual environment or Conda.
#'   Default is `FALSE`. If both `use_system_python = TRUE` and `use_conda = TRUE`, the function will prioritize system Python.
#' @param create_if_missing Logical. If `TRUE`, the function creates the specified virtual environment (`envname`) if it does not exist.
#'   Default is `FALSE`, meaning it will only warn if `envname` is missing and list available environments.
#'   This applies only when `use_conda = FALSE`, as Conda environments must be created manually.
#' 
#' @return This function configures the Python environment but does not return any value.
#' @details
#' \itemize{
#'         \item The function automatically detects the best available Python environment.
#'         \item If a user specifies `envname`, it tries to activate that environment.
#'         \item If both `use_system_python = TRUE` and `use_conda = TRUE`, the function will prioritize system Python.
#'         \item It ensures required Python packages (`torch`, `numpy`, `pandas`, `tqdm`) are available using `py_require()`.
#'         \item It is recommended to set up the `reticulate` package properly before running `setup_lbcnet()`.
#'         \item If encountering errors like `"not a Python virtualenv"`, it is advised to delete and recreate the virtual environment.
#'         \item If multiple Python versions exist on the system, ensure that packages are installed in the correct Python environment.
#'         \item Use `reticulate::py_config()` to verify the active Python environment before running the function.
#'       }
#'
#' @examples
#' \dontrun{
#' setup_lbcnet()  # Automatically configures the best available Python environment
#' setup_lbcnet(envname = "r-lbcnet")  # Uses a specific virtual environment (warns if missing)
#' setup_lbcnet(envname = "r-lbcnet", create_if_missing = TRUE)  # Creates "r-lbcnet" if missing
#' setup_lbcnet(use_conda = TRUE)  # Forces Conda if available
#' setup_lbcnet(use_system_python = TRUE)  # Forces to use system Python
#' setup_lbcnet(use_system_python = TRUE, use_conda = TRUE)  # Prioritizes system Python over Conda
#' }
#'
#' @importFrom reticulate use_python use_virtualenv use_condaenv py_config py_module_available py_install
#' @export
setup_lbcnet <- function(use_conda = FALSE, envname = "r-lbcnet", use_system_python = FALSE, create_if_missing = FALSE) {
  message("LBCNet: Configuring Python environment...")
  
  required_modules <- c("torch", "numpy", "pandas", "tqdm")
  
  # Check if Python is already initialized
  if (reticulate::py_available(initialize = FALSE)) {
    current_python <- reticulate::py_config()$python
    message("Python is already initialized: ", current_python)
  
    ephemeral_patterns <- c("reticulate/uv/cache", "reticulate/virtualenvs/cache")
    if (any(sapply(ephemeral_patterns, function(p) grepl(p, current_python, fixed = TRUE)))) {
      
      message("Detected ephemeral virtual environment.")
      message("This environment resets every R session and may cause issues.")
      
      message("Removing ephemeral virtual environment...")
      unlink(tools::R_user_dir("reticulate", "cache"), recursive = TRUE, force = TRUE)
      unlink(tools::R_user_dir("reticulate", "data"), recursive = TRUE)
      
      stop("An ephemeral virtual environment was detected and removed. Please restart your R session and run `setup_lbcnet()` again.", call. = FALSE)
    }
    
    message("Using the existing Python environment instead of: ", envname)
    
    missing_modules <- required_modules[!sapply(required_modules, reticulate::py_module_available)]
    if (length(missing_modules) > 0) {
      message("Installing missing Python packages: ", paste(missing_modules, collapse = ", "))
      
      if (reticulate::py_module_available("pip")) {
        if (grepl("virtualenvs", reticulate::py_config()$python)) {
          # Virtual environment: Use py_install()
          reticulate::py_install(missing_modules)
        } else {
          # System Python: Use system pip
          system(paste(shQuote(current_python), "-m pip install --upgrade", paste(missing_modules, collapse = " ")))
        }
      } else {
        stop("pip` is not available in the current Python environment. Please install it manually.")
      }
    } else {
      message("All required Python packages are already installed.")
    }
    
    # Confirm successful configuration
    message("LBCNet is using Python from: ", reticulate::py_config()$python)
    
    return(invisible(NULL))
  } 
  
  # Case 1: User explicitly chooses system Python
  # Define system Python path
  system_python <- Sys.which("python")
  if (use_system_python) {
    if (nzchar(system_python)) {
      reticulate::use_python(system_python, required = TRUE)
      message("Using system Python: ", system_python)
    } else {
      stop("No system Python found! Please install Python and configure reticulate.")
    }
    return(invisible(NULL))  # Exit the function since the user has chosen system Python.
  }
  
  message("No active Python environment detected. Detecting available environments...")
  valid_virtualenvs <- reticulate::virtualenv_list()

  # Case 2: Use Python from RETICULATE_PYTHON if set
  if (nzchar(Sys.getenv("RETICULATE_PYTHON", unset = ""))) {
    reticulate::use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
    message("Using RETICULATE_PYTHON: ", Sys.getenv("RETICULATE_PYTHON"))
  
  # Case 3: Use Conda if requested and available
  } else if (use_conda && reticulate::conda_binary() != "") {
    reticulate::use_condaenv(envname, required = TRUE)
    message("Using Conda environment: ", envname)
    
  # Case 4: Use an existing virtual environment
  } else if (envname %in% valid_virtualenvs) {
    reticulate::use_virtualenv(envname, required = TRUE)
    message("Using Virtual Environment: ", envname)
  
  } else {
    message("Warning: The specified virtual environment '", envname, "' does not exist. Please check the name.")
    message("Available virtual environments: ", paste(valid_virtualenvs, collapse = ", "))
    
    if (create_if_missing) {
      message("Creating a new virtual environment: ", envname)
      reticulate::virtualenv_create(envname)
      reticulate::use_virtualenv(envname, required = TRUE)
    } else {
      message("Virtual environment '", envname, "' was not found. Falling back to system Python.")
      
      # Case 5: Fall back to system Python if no other options are available
      if (nzchar(system_python)) {
        reticulate::use_python(system_python, required = TRUE)
        message("Using system Python: ", system_python)
      } else {
        stop("No Python environment found! Please install Python and configure reticulate.")
      }
    }
  } 
  
  current_python <- reticulate::py_config()$python
  message("LBCNet is using Python from: ", current_python)
  
  # Install missing Python packages
  missing_modules <- required_modules[!sapply(required_modules, reticulate::py_module_available)]
  if (length(missing_modules) > 0) {
    message("Installing missing Python packages: ", paste(missing_modules, collapse = ", "))
    
    if (reticulate::py_module_available("pip")) {
      if (grepl("virtualenvs", current_python)) {
        # Virtual environment: Use py_install()
        reticulate::py_install(missing_modules)
      } else {
        # System Python: Use system("pip install ...") but only for missing packages
        system(paste("pip install", paste(missing_modules, collapse = " ")))
      }
    } else {
      stop("pip` is not available in the current Python environment. Please install it manually.")
    }
  } else {
    message("All required Python packages are already installed.")
  }
  
  # Confirm successful configuration
  message("LBCNet is using Python from: ", reticulate::py_config()$python)
}
