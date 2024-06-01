# Load the packages
library(readr) #read_csv
library(VIM) #knn
library(dplyr)


# Read the CSV file
dataset <- read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
str(dataset)

# Count the number of NAs in each column
na_counts <- colSums(is.na(dataset))
print("NA counts in each column:")
print(na_counts) #TotalCharges: 11

## Delete the 'email' column
##dataset <- subset(dataset, select = -customerID)

# Perform KNN imputation
imputed_data <- kNN(dataset, k = 3, variable = c("TotalCharges"), imp_var = FALSE)

# Standardize text
Standardized_data <- imputed_data %>%
  mutate(across(
    .cols = c("MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaymentMethod"),  # Specify the columns to transform
    .fns = ~ ifelse(grepl("^N", .), "No", .)  # Define the transformation function with pattern matching
  ))
print(Standardized_data)


# View the changes
sum(is.na(Standardized_data))
str(Standardized_data)
print(Standardized_data)


# Save the dataset to a CSV file
write.csv(Standardized_data, "data/clean/imputed_dataset.csv", row.names = FALSE)

