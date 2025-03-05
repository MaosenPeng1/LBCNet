#' Setup Python Environment for LBCNet
#'
#' @description This function configures the Python environment for LBCNet, ensuring the correct
#' Python version and dependencies are loaded. It prioritizes virtual environments (`venv`) over
#' Conda unless explicitly requested.
#'
#' @param use_conda Logical. If `TRUE`, it attempts to use Conda (`r-lbcnet`) instead of virtualenv.
#'   Default is `FALSE`, meaning virtualenv/system Python is preferred.
#' @param envname A character string specifying the name of the virtual environment or Conda environment
#'   to use.  Default is `"r-lbcnet"`. One can use `virtualenv_list()` or `conda_list()` to check check the available Python environments in your system.
#'
#' @return This function configures the Python environment but does not return any value.
#' @details
#' - The function automatically detects the best available Python environment.
#' - If a user specifies `env`, it tries to activate that environment.
#' - It ensures required Python packages (`torch`, `numpy`, `pandas`, `tqdm`) are available using `py_require()`.
#' - If `RETICULATE_PYTHON` is set in `.Renviron`, it will respect that setting.
#'
#' @examples
#' \dontrun{
#' setup_lbcnet()  # Automatically configures the best available Python environment
#' setup_lbcnet(envname = "myenv")  # Uses a specific virtual environment
#' setup_lbcnet(use_conda = TRUE)  # Forces Conda if available
#' }
#'
#' @importFrom reticulate use_python use_virtualenv use_condaenv py_config py_module_available py_install
#' @export
setup_lbcnet <- function(use_conda = FALSE, envname = "r-lbcnet") {
  message("LBCNet: Configuring Python environment...")

  # Detect if an ephemeral environment is being used
  ephemeral_path <- tools::R_user_dir("reticulate", "cache")
  if (grepl(ephemeral_path, py_config()$python, fixed = TRUE)) {
    message("Warning: Reticulate is using an ephemeral virtual environment.")
    message("   This environment resets in every R session and may cause issues.")

    choice <- readline("Do you want to create a persistent virtual environment instead? (y/n): ")
    if (tolower(choice) == "y") {
      message("Removing ephemeral virtual environment...")

      # Remove ephemeral cache directories
      unlink(tools::R_user_dir("reticulate", "cache"), recursive = TRUE, force = TRUE)
      unlink(tools::R_user_dir("reticulate", "data"), recursive = TRUE)

      message("Creating a persistent virtual environment: ", envname)
      reticulate::virtualenv_create(envname)
    } else {
      message("Proceeding with the ephemeral environment. Issues may occur.")
    }
  }

  # Automatically detect the best Python environment
  if (nzchar(Sys.getenv("RETICULATE_PYTHON", unset = ""))) {
    reticulate::use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
    message("Using RETICULATE_PYTHON: ", Sys.getenv("RETICULATE_PYTHON"))
  } else if (use_conda && reticulate::conda_binary() != "") {
    reticulate::use_condaenv(envname, required = TRUE)
    message("Using Conda environment: ", envname)
  } else if (envname %in% reticulate::virtualenv_list()) {
    reticulate::use_virtualenv(envname, required = TRUE)
    message("Using Virtual Environment: ", envname)
  } else {
    system_python <- Sys.which("python")
    if (nzchar(system_python)) {
      reticulate::use_python(system_python, required = TRUE)
      message("Using system Python: ", system_python)
    } else {
      stop("No Python environment found! Please install Python and configure reticulate.")
    }
  }

  # Install missing Python packages
  required_modules <- c("torch", "numpy", "pandas", "tqdm")
  missing_modules <- required_modules[!sapply(required_modules, reticulate::py_module_available)]

  if (length(missing_modules) > 0) {
    message("Installing missing Python packages: ", paste(missing_modules, collapse = ", "))
    reticulate::py_install(missing_modules, envname = reticulate::py_config()$python)
  }

  # Confirm successful configuration
  message("LBCNet is using Python from: ", reticulate::py_config()$python)
}
