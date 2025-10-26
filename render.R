if (!requireNamespace("rmarkdown", quietly = TRUE)) install.packages("rmarkdown")
rmarkdown::render("report.Rmd", output_format = "html_document")
cat("Report rendered to HTML.\n")


