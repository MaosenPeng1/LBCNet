.onLoad <- function(libname, pkgname) {
  library(reticulate)

  message("LBCNet: Configuring Python environment...")

  # Automatically detect the best available Python environment
  if (nzchar(Sys.getenv("RETICULATE_PYTHON", unset = ""))) {
    python_path <- Sys.getenv("RETICULATE_PYTHON")
    use_python(python_path, required = TRUE)
    message("✔ Using RETICULATE_PYTHON: ", python_path)
  } else {
    # Check for Conda
    conda_available <- tryCatch(
      !is.null(conda_binary()),
      error = function(e) FALSE  # If an error occurs, assume Conda is missing
    )

    if (conda_available) {
      # Use Conda environment if available
      conda_env <- "r-lbcnet"
      use_condaenv(conda_env, required = FALSE)
      message("✔ Using Conda environment: ", conda_env)
    } else if (length(virtualenv_list()) > 0) {
      # Use Virtual Environment if Conda is missing
      venv <- "r-reticulate"
      use_virtualenv(venv, required = FALSE)
      message("✔ Using Virtual Environment: ", venv)
    } else {
      # Last fallback: Use system Python
      system_python <- Sys.which("python")
      if (nzchar(system_python)) {
        use_python(system_python, required = TRUE)
        message("✔ Using system Python: ", system_python)
      } else {
        stop("❌ No Python environment found! Please install Python and configure reticulate.")
      }
    }
  }

  # List of required Python modules
  required_modules <- c("torch", "numpy", "pandas", "tqdm")

  # Install missing Python modules
  missing_modules <- required_modules[!py_module_available(required_modules)]
  if (length(missing_modules) > 0) {
    message("⚠️ Installing missing Python packages: ", paste(missing_modules, collapse = ", "))
    py_install(missing_modules)
  }

  # Confirm successful configuration
  message("✔ LBCNet is using Python from: ", py_config()$python)
}
