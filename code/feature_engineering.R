library(readr)
library(caret)
library(ROSE)
library(car)

### Read the CSV file
data <- read_csv("data/clean/imputed_dataset.csv")
str(data)

### Drop TotalCharges:tenure and MonthlyCharges are highly correlated with the column TotalCharges. (MonthlyCharges multiply tenure)
model_with_totalcharges <- lm(TotalCharges ~ MonthlyCharges + tenure, data=data)
summary(model_with_totalcharges)
vif(model_with_totalcharges) 

model_without_totalcharges <- lm(MonthlyCharges ~ tenure, data=data)
summary(model_without_totalcharges)

anova(model_with_totalcharges, model_without_totalcharges)

data <- data[, -which(names(data) == "TotalCharges")]

### Label encoding
columns_to_convert <- c("gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn")
exclude_cols <- c("customerID", "Contract", "PaymentMethod")
cols_to_use <- setdiff(names(data), exclude_cols)

# Custom mapping for InternetService
custom_mapping <- function(column) {
  mapping <- c('No' = 0, 'DSL' = 1, 'Fiber optic' = 2)
  return(mapping[column])
}
# Apply the custom mapping to the InternetService column
data$InternetService <- custom_mapping(data$InternetService)

char_columns <- sapply(data[, cols_to_use], is.character)
data[, cols_to_use][char_columns] <- lapply(data[, cols_to_use][char_columns], function(x) as.numeric(as.factor(x))- 1)
LabelEncoded_data<-data
str(LabelEncoded_data)

###One-hot encoding
PaymentMethod_OneHot_encoded <- model.matrix(~PaymentMethod - 1, data = LabelEncoded_data)
PaymentMethod_OneHot_encoded <- as.data.frame(PaymentMethod_OneHot_encoded)
print(PaymentMethod_OneHot_encoded)

Contract_OneHot_encoded <- model.matrix(~ Contract - 1, data = LabelEncoded_data)
Contract_OneHot_encoded <- as.data.frame(Contract_OneHot_encoded)
print(Contract_OneHot_encoded)

#Find the original positions of the "PaymentMethod" and "Contract" columns
contract_index <- which(names(LabelEncoded_data) == "Contract")
pm_index <- which(names(LabelEncoded_data) == "PaymentMethod")

# Remove the original"Contract" and "PaymentMethod" columns
LabelEncoded_data <- LabelEncoded_data[, !names(LabelEncoded_data) %in% c("Contract", "PaymentMethod")]

# Convert the data frame to a list to insert columns more flexibly
LabelEncoded_data_list <- as.list(LabelEncoded_data)

# Insert the one-hot encoded columns at the original positions
LabelEncoded_data_list <- append(LabelEncoded_data_list, as.list(Contract_OneHot_encoded[, grep("Contract", names(Contract_OneHot_encoded))]), after = contract_index -1)
LabelEncoded_data_list <- append(LabelEncoded_data_list, as.list(PaymentMethod_OneHot_encoded[, grep("PaymentMethod", names(PaymentMethod_OneHot_encoded))]), after = pm_index + 1)

# Convert the list back to a data frame
encoded_data <- as.data.frame(LabelEncoded_data_list)

print(encoded_data)

###Oversampling
# class_distribution <- table(encoded_data$Churn)
# print(class_distribution)

# Perform random oversampling
# oversampled_data <- ovun.sample(Churn ~ ., data = encoded_data, method = "over", seed = 123)
# oversampled_data <- oversampled_data$data  # Access oversampled data directly
# table(oversampled_data$Churn)
# print(oversampled_data)


### Scaling numeric feature
# Columns to scale
numeric_columns <- c("tenure", "MonthlyCharges")

# Range scaling (min-max scaling)
#preProcess_scale <- preProcess(oversampled_data[, numeric_columns], method = "range", rangeBounds = c(0, 1))
#oversampled_data[, numeric_columns] <- predict(preProcess_scale, oversampled_data[, numeric_columns])

preProcess_scale <- preProcess(encoded_data[, numeric_columns], method = "range", rangeBounds = c(0, 1))
encoded_data[, numeric_columns] <- predict(preProcess_scale, encoded_data[, numeric_columns])

# Standardization
standardize <- function(x) {
  return(scale(x))
}

# Apply standardization to the specified columns
#oversampled_data[, numeric_columns] <- lapply(oversampled_data[, numeric_columns], standardize)
encoded_data[, numeric_columns] <- lapply(encoded_data[, numeric_columns], standardize)

# Assign the standardized data to Normalizing_data
# standardized_data <- oversampled_data
standardized_data <- encoded_data
str(standardized_data)

# Save the engineered dataset to a CSV file
write.csv(standardized_data, "data/feature/feature_set.csv", row.names = FALSE)

