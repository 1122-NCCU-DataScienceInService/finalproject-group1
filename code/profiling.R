# Load the readr package
library(readr)

# Read the CSV file
dataset <- read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Load the DataExplorer package
library(DataExplorer)

# Generate a profile report
# profile_report <- create_report(dataset)

# Save the report to a file
create_report(dataset, 
              y = "Churn",
              config = configure_report(
                plot_correlation_args = list("cor_args" = list("use" = "pairwise.complete.obs"))
              ),
              output_file = "profiling.html",
              output_dir = 'docs')
