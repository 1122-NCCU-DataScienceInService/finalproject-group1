# 1. Read the dataset
library(readr)
data <- read_csv("./data/feature/feature_set.csv")

library(ggplot2)
library(lattice)
library(caret)
library(xgboost)
library(pROC)
library(dplyr)

# 2. Separate the data into training and testing set
n <- nrow(data)
data <- data[sample(n),]  #將資料進行隨機排列
index <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
train_data <- data[index, ] 
test_data <- data[-index, ]

# 3. Using training set to train the xgboost model 
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[-c(1, which(names(train_data) == "Churn"))]), label = train_data$Churn)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[-c(1, which(names(test_data) == "Churn"))]), label = test_data$Churn)

# 4. k-fold cross validation
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6
)
nrounds <- 100
nfold <- 5
early_stopping_rounds <- 10
verbose <- 1

# Create DMatrix object
#data_matrix <- xgb.DMatrix(data = as.matrix(data[-c(1, which(names(data) == "Churn"))]), label = data$Churn)

# using xgb.cv to do k-fold cross validation 
cv_result <- xgb.cv(
  params = params,
  data = train_matrix,
  nrounds = nrounds,
  nfold = nfold,
  early_stopping_rounds = early_stopping_rounds,
  verbose = verbose
)

print(cv_result)

# get the best iteration
best_nrounds <- cv_result$best_iteration

model <- xgb.train(
  params = params,
  data = data_matrix,
  nrounds = best_nrounds,
  verbose = TRUE
)

# 5. Predict label and probability
train_pred <- predict(model, train_matrix)
train_predicted_label <- ifelse(train_pred > 0.5, 1, 0)
test_pred <- predict(model, test_matrix)
test_predicted_label <- ifelse(test_pred > 0.5, 1, 0)

# 6. Compute the ROC of training and testing data
roc_train <- roc(train_data$Churn, train_pred)
roc_test <- roc(test_data$Churn, test_pred)
auc_train <- auc(roc_train)
auc_test <- auc(roc_test)
print(paste("AUC of training data: ", auc_train))
print(paste("AUC of testing data: ", auc_test))

# 7. Null model comparison
# Null model prediction: using the mean of Churn in the training set as the probability
mean_churn <- mean(train_data$Churn)
null_train_pred <- rep(mean_churn, nrow(train_data))
null_test_pred <- rep(mean_churn, nrow(test_data))

# Compute the ROC of null model
roc_null_train <- roc(train_data$Churn, null_train_pred)
roc_null_test <- roc(test_data$Churn, null_test_pred)
auc_null_train <- auc(roc_null_train)
auc_null_test <- auc(roc_null_test)

# Print AUC values for the null model
print(paste("AUC for null model on training data: ", auc_null_train))
print(paste("AUC for null model on testing data: ", auc_null_test))

# 使用 DeLong 檢驗比較兩個 AUC
roc_test_xgb <- roc(test_data$Churn, test_pred)
roc_test_null <- roc(test_data$Churn, null_test_pred)

# DeLong 檢驗
delong_test <- roc.test(roc_test_xgb, roc_test_null, method="delong")

# 輸出檢驗結果
print(delong_test)

# 8. Feature importance and ranking
importance_matrix <- xgb.importance(model = model, feature_names = colnames(train_data[-c(1, which(names(train_data) == "Churn"))]))
xgb.plot.importance(importance_matrix)

# 9. Save the prediction result
train_results <- data.frame(customerID = train_data$customerID, label = train_predicted_label, probability = train_pred, groundtruth = train_data$Churn)
test_results <- data.frame(customerID = test_data$customerID, label = test_predicted_label, probability = test_pred, groundtruth = test_data$Churn)
write_csv(train_results, "./outputs/train_predictions.csv")
write_csv(test_results, "./outputs/test_predictions.csv")

# 10. Save the feature importance result
write_csv(importance_matrix, "./outputs/feature_importance.csv")

# 11. Save the XGBoost model
xgb.save(model, "./model/churn_prediction_model.xgb")

# 12. Save the ROC curve
# Assuming roc_train_plot and roc_test_plot are ggplot objects for ROC curves
roc_train_plot <- ggroc(roc_train) + ggtitle("ROC Curve - Training Data")
roc_test_plot <- ggroc(roc_test) + ggtitle("ROC Curve - Testing Data")

# Save ROC curves
ggsave("./outputs/roc_train.png", plot = roc_train_plot)
ggsave("./outputs/roc_test.png", plot = roc_test_plot)

# 13. Lift analysis of the prediction result
# sorting by probability
test_results <- test_results[order(-test_results$probability),]

# Segmenat the customer
test_results$decile <- cut(test_results$probability, breaks=quantile(test_results$probability, probs=seq(0, 1, by = 0.1)), include.lowest=TRUE, labels=FALSE)

# Reverse the decilne numbering
test_results$decile <- 11 - test_results$decile

# 計算每個分組的實際響應率和 Lift
test_lift_df <- test_results %>%
  group_by(decile) %>%
  summarise(
    count = n(),
    num_responses = sum(label),
    response_rate = mean(label),
    lift = response_rate / mean(data$Churn)
  )

plot <- ggplot(test_lift_df, aes(x = as.factor(decile), y = lift)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Lift Chart", x = "Decile", y = "Lift") +
  theme_minimal()

# 使用 ggsave 保存圖形
ggsave("./outputs/lift_chart.png", plot, width = 10, height = 6, dpi = 300)

# Save the lift data to CSV
write.csv(test_lift_df, "./outputs/lift_data.csv", row.names = FALSE)

