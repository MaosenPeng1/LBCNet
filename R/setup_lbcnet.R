#' Set Up Python Environment for LBCNet
#'
#' @description
#' This function configures the Python environment for the LBCNet package.
#' It automatically detects and sets the best available Python environment
#' (Conda, virtualenv, or system Python) and ensures that all required
#' Python dependencies (`torch`, `numpy`, `pandas`, `tqdm`) are installed.
#'
#' This function should be run only once after installation to ensure the proper setup.
#'
#' @details
#' \itemize{
#'         \item If a Python environment is already specified via `RETICULATE_PYTHON`, it will use that.
#'         \item If Conda is installed, it will attempt to use the `"r-lbcnet"` Conda environment.
#'         \item If a virtual environment (`r-reticulate`) exists, it will be used.
#'         \item Otherwise, it will use the system Python if available.
#'         \item If no valid Python is found, it will prompt the user to install Python manually.
#'       }
#'
#' Python Package Installation:
#' - If any required Python modules (`torch`, `numpy`, `pandas`, `tqdm`) are missing,
#'   they will be installed automatically.
#'
#' @return
#' This function does not return any value. Instead, it sets up the Python environment
#' for the package and ensures all dependencies are installed.
#'
#' @examples
#' \dontrun{
#' # Run this once after installing LBCNet
#' setup_lbcnet()
#' }
#'
#' @importFrom reticulate py_finalize use_python use_condaenv use_virtualenv conda_binary virtualenv_list py_module_available py_install py_config py_discover_config
#' @export
setup_lbcnet <- function() {
  library(reticulate)

  message("LBCNet: Configuring Python environment...")

  # Unset Python session to force reconfiguration (only if needed)
  if (reticulate::py_available()) {
    reticulate::py_finalize()
  }

  # Automatically detect the best available Python environment
  if (nzchar(Sys.getenv("RETICULATE_PYTHON", unset = ""))) {
    python_path <- Sys.getenv("RETICULATE_PYTHON")
    reticulate::use_python(python_path, required = TRUE)
    message("✔ Using RETICULATE_PYTHON: ", python_path)
  } else {
    # Check if Conda is available safely
    conda_available <- tryCatch(
      !is.null(reticulate::conda_binary()),
      error = function(e) FALSE
    )

    if (conda_available) {
      conda_env <- "r-lbcnet"
      reticulate::use_condaenv(conda_env, required = FALSE)
      message("✔ Using Conda environment: ", conda_env)
    } else if (length(reticulate::virtualenv_list()) > 0) {
      venv <- "r-reticulate"
      reticulate::use_virtualenv(venv, required = FALSE)
      message("✔ Using Virtual Environment: ", venv)
    } else {
      system_python <- Sys.which("python")
      if (nzchar(system_python)) {
        reticulate::use_python(system_python, required = TRUE)
        message("✔ Using system Python: ", system_python)
      } else {
        stop("❌ No Python environment found! Please install Python and configure reticulate.")
      }
    }
  }

  # Install missing Python packages
  required_modules <- c("torch", "numpy", "pandas", "tqdm")
  missing_modules <- required_modules[!reticulate::py_module_available(required_modules)]
  if (length(missing_modules) > 0) {
    message("⚠️ Installing missing Python packages: ", paste(missing_modules, collapse = ", "))
    reticulate::py_install(missing_modules)
  }

  # Confirm successful configuration
  message("✔ LBCNet is using Python from: ", reticulate::py_config()$python)
}
