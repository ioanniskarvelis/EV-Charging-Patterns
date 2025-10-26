required_packages <- c(
  "lubridate", "ggplot2", "gridExtra", "moments", "corrplot", "caret",
  "glmnet", "randomForest", "xgboost", "keras", "car", "dplyr",
  "tibble", "purrr", "here", "patchwork", "viridis"
)

is_installed <- function(pkg) {
  isTRUE(pkg %in% rownames(installed.packages()))
}

to_install <- required_packages[!vapply(required_packages, is_installed, logical(1))]

if (length(to_install)) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

message("All required packages are installed.")


