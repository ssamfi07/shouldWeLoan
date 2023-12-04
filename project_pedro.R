library(tidyverse)
library(caret)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggbiplot)
library(factoextra)
library(stats)
library(cluster)
library(naivebayes)
library(class)
library(rpart)
library(rpart.plot)
library(randomForest)
library(dplyr)
library(glmnet)

#read the csv file
df <- read_csv("database.csv") 
print(colnames(df))
view(df)

#restart R if necessary
#.rs.restartR()

#TASK1: DATA UNDERSTANDING AND PREPARATION------------------------------------------------------------

#DATA CLEANING

#we don't need "account_id", "trans_id", "date_diff"
df <- df[, !colnames(df) %in% c("account_id", "trans_id", "date_diff")]
#convert data_diff to numeric
#df$date_diff <- as.numeric(df$date_diff) 

#check for missing values
#colSums(is.na(df))
#we have missing values in the columns "date_diff"
#so we need to change the N/A values to "0"
#df$date_diff[is.na(df$date_diff)] <- 0
#colSums(is.na(df))

#check for redundant or irrelevant values (or columns)
#check for columns with 0 variance
#numeric_df <- df[sapply(df, is.numeric)]
#zero_variance_cols <- colnames(numeric_df)[apply(numeric_df, 2, function(x) length(unique(x))) == 1]
#print(zero_variance_cols)
#no column has 0 variance

#DATA TRANSFORMATION

#normalize every numeric column (with z-score standardization), but the target variable "status"
numeric_cols <- sapply(df, is.numeric)
numeric_cols["loan_id"] <- FALSE
numeric_cols["status"] <- FALSE
numeric_cols["type"] <- FALSE
df[numeric_cols] <- scale(df[numeric_cols])

#check for categorical columns
categorical_cols <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
print(categorical_cols)
#only the column "type" is categorical
#so we are going to label encode it
df <- df %>% mutate(type = recode(type, "credit" = 1, "withdrawal" = 2, "withdrawal in cash" = 3))

#convert "status" to 1s and 0s
df$status<- ifelse(df$status == 1, 1, 0)

#DATA REDUCTION

#(Pearson) correlation in relation to "status" (our target variable)
correlations <- sapply(df, function(col) cor(col, df$status))
print(correlations)
#the columns that have almost no correlation with "status" will be dropped 
#only "date_diff" will be dropped
#df <- df[, !colnames(df) %in% c("date_diff")]

#PCA in relation to "account_id"
pca <- prcomp(df, scale = TRUE)
summary(pca)
#visual representation of the PCA (1 and 2)
#biplot(pca, scale = 0, choices = c(1, 2))
#new dataframe with only 2 PCs
pcs_only <- as.data.frame(pca$x[, 1:2])

#TASK2: DESCRIPTIVE MODELLING-------------------------------------------------------------------------

#HIERARCHICAL CLUSTERING

#with euclidean and ward method
dist_matrix <- dist(pcs_only, method = "euclidean")
hclust_model <- hclust(dist_matrix, method = "ward.D2")
#plot(hclust_model, hang = 0.1)
#fviz_dend(hclust_model, k = 3, cex = 0.6, main = "Dendrogram of Hierarchical Clustering")
#not readable


#PARTITIONAL CLUSTERING

#graph with the within cluster sum of squares to determine the ideal number of clusters
wcss <- vector()
for (i in 1:10) {
  kmeans_model <- kmeans(pcs_only , centers = i, nstart = 10)
  wcss[i] <- kmeans_model$tot.withinss
}
plot(1:10, wcss, type = "b", xlab = "Number of Clusters", ylab = "WCSS")
diff_wcss <- c(0, diff(wcss))
#we apply the elbow method, and the ideal number of clusters is 3 or 4

#we will apply partitional clustering with 3 and 4 clusters and determine which one is better
num_clusters <- 4
cluster_cut <- cutree(hclust_model, k = num_clusters)
#adds a new column to the dataframe with the clusters id
pcs_only$Cluster <- as.factor(cluster_cut)
#visualize the clusters
ggplot(pcs_only, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point() +
  labs(title = "Cluster Analysis", x = "PC1", y = "PC2")

#silhouette coefficient to measure how good the partitional clustering is
sf <- silhouette(cluster_cut, dist(pcs_only))
sf_avg <- mean(sf[, 3])
print(sf_avg)
#0.46

#k-means (to numeric columns)
numeric_cols <- sapply(pcs_only, is.numeric)
pcs_only_numeric <- pcs_only[, numeric_cols]
kmeans_model <- kmeans(pcs_only, centers = num_clusters, nstart = 25)
#visualize the clusters from k-means
fviz_cluster(kmeans_model, data = pcs_only_numeric, geom = "point",
             ellipse.type = "t", ggtheme = theme_classic(),
             ellipse.alpha = 0.5, palette = "jco",
             main = "K-means Clustering Results",
             xlab = "PC1", ylab = "PC2")

#silhouette coefficient to measure how good the k-means is
sf_kmeans <- silhouette(kmeans_model$cluster, dist(pcs_only_numeric))
sf_kmeans_avg <- mean(sf_kmeans[,3])
print(sf_kmeans_avg) 
#0.33

#4 clusters seems to be the ideal number of clusters


#TASK 3: PREDICTIVE MODELLING-------------------------------------------------------------------------

#SPLIT THE DATA

split_data <- function(df) {
  #random split
  set.seed(123)
  #80% training and 20% testing
  train_indices <- sample(1:nrow(df), 0.8*nrow(df))
  #training dataframe
  train_data <- df[train_indices, ]
  #testing dataframe
  test_data <- df[-train_indices, ]
  
  return(list(train_data = train_data, test_data = test_data))
}

#split the data of the original dataframe
view(df)
split_df <- split_data(df)
train_data_split_df <- split_df$train_data
test_data_split_df <- split_df$test_data

#split the data of the dataframe with the first 2 PCs and the cluster
pcs_only <- pcs_only %>% mutate(status = df$status)
pcs_only <- cbind(pcs_only, loan_id = df$loan_id)
view(pcs_only)
split_pcs_only <- split_data(pcs_only)
train_data_split_pcs_only <- split_pcs_only$train_data
test_data_split_pcs_only <- split_pcs_only$test_data

#split the data of the dataframe with the original dataframe, the first 2 PCs and the cluster
pcs_only <- pcs_only[, !(names(pcs_only) %in% c("status", "loan_id"))]
df_pcs_cluster <- cbind(df, pcs_only)
view(df_pcs_cluster)
split_df_pcs_cluster <- split_data(df_pcs_cluster)
train_data_split_df_pcs_cluster <- split_df_pcs_cluster$train_data
test_data_split_df_pcs_cluster <- split_df_pcs_cluster$test_data
pcs_only <- pcs_only %>% mutate(status = df$status)
pcs_only <- cbind(pcs_only, loan_id = df$loan_id)

#TEST

testing <- function(test_data) {
  #calculating accuracy
  correct_predictions <- sum(test_data$predicted == test_data$status)
  total_predictions <- length(test_data$status)
  accuracy <- correct_predictions / total_predictions
  print(paste("Accuracy:", accuracy))
  
  #calculating error rate
  error_rate <- 1 - accuracy
  print(paste("Error rate:", error_rate))
  
  return(list(accuracy = accuracy, error_rate = error_rate))
}

testing2 <- function(df) {
  #calculating accuracy
  accuracy <- mean(df$predicted_binary == df$status)
  print(paste("Accuracy:", accuracy))
  
  #calculating error rate
  error_rate <- 1 - accuracy
  print(paste("Error rate:", error_rate))
}

testing3 <- function(test_data) {
  #calculating accuracy
  accuracy <- mean(test_data$predicted_binary == test_data$status)
  print(paste("Accuracy:", accuracy))
  
  #calculating error rate
  error_rate <- 1 - accuracy
  print(paste("Error rate:", error_rate))
}

testing4 <- function(df) {
  #calculating accuracy
  accuracy <- mean(predicted == df$status)
  cat("Accuracy:", accuracy, "\n")
  
  
  #calculating error rate
  error_rate <- 1 - accuracy
  cat("Error rate:", error_rate, "\n")
}

#NAIVE BAYES

naive_bayes_model <- function(train_data, test_data) {
  #training all columns except our target variable "status" (we need to convert it to a factor)
  train_data$status <- as.factor(train_data$status)
  naive_model <- naive_bayes(train_data[, -which(names(train_data) == "status")], train_data$status)
  
  #testing  all columns except our target variable "status"
  test_data$predicted <- predict(naive_model, newdata = test_data[, -which(names(test_data) == "status")])
  
  test_data <- testing(test_data)
  return(test_data)
}

#original dataframe
test_naive_bayes_model <- naive_bayes_model(train_data_split_df, test_data_split_df)
#Accuracy - 0.86 / Error rate - 0.14
#dataframe with the first 2 PCs and the cluster
test_naive_bayes_model <- naive_bayes_model(train_data_split_pcs_only, test_data_split_pcs_only)
#Accuracy - 0.85 / Error rate - 0.15
#dataframe with the original dataframe, the first 2 PCs and the cluster
test_naive_bayes_model <- naive_bayes_model(train_data_split_df_pcs_cluster, test_data_split_df_pcs_cluster)
#Accuracy - 0.81 / Error rate - 0.18


#KNN


#LOGISTIC REGRESSION

logistic_regression_model <- function(train_data, test_data, df) {
  #apply our model
  logistic_model <- glm(df$status ~ ., data = df[, !(names(df) %in% c("status"))], family = "binomial")
 
  #make predictions
  df$predicted <- predict(logistic_model, type = "response")
  df$predicted_binary <- ifelse(df$predicted >= 0.5, 1, 0)
  
  test_data <- testing2(df)
  
  return(test_data)
}

#original dataframe
test_logistic_regression_model <- logistic_regression_model(train_data_split_df, test_data_split_df, df)
#Accuracy - 0.88 / Error rate - 0.12
#dataframe with the first 2 PCs and the cluster
test_logistic_regression_model <- logistic_regression_model(train_data_split_pcs_only, test_data_split_pcs_only, pcs_only)
#Accuracy - 0.89 / Error rate - 0.11 
#dataframe with the original dataframe, the first 2 PCs and the cluster
test_logistic_regression_model <- logistic_regression_model(train_data_split_df_pcs_cluster, test_data_split_df_pcs_cluster, df_pcs_cluster)
#Accuracy - 1 / Error rate - 0
#to much information and shows perfect separation
#so we are going to add feature selection and regularization

logistic_regression_model2 <- function(train_data, test_data, df) {
  #feature selection
  threshold_value = 0.1
  high_variance_cols <- apply(df, 2, function(x) var(x) > threshold_value)
  df <- df[, high_variance_cols]
  
  #regularization
  predictors <- df[, !(names(df) %in% c("status"))]
  target <- df$status
  
  #apply our model
  logistic_model <- glmnet(as.matrix(predictors), target, family = "binomial", alpha = 1)
  
  #make predictions
  test_data$predicted <- predict(logistic_model, newx = as.matrix(test_data[, !(names(test_data) %in% c("status"))]), type = "response")
  test_data$predicted_binary <- ifelse(test_data$predicted >= 0.5, 1, 0)
  
  test_data <- testing3(test_data)
  
  return(test_data)
}

#dataframe with the original dataframe, the first 2 PCs and the cluster
test_logistic_regression_model2 <- logistic_regression_model2(train_data_split_df_pcs_cluster, test_data_split_df_pcs_cluster, df_pcs_cluster)
#Accuracy - 0.97 / Error rate - 0.03


#DECISION TREES

#apply our model
decision_tree_model <- function(df){
  #apply our model
  tree_model <- rpart(df$status ~ ., data = df[, !(names(df) %in% c("status"))], method = "class")

  #make predictions
  predicted <- predict(tree_model, type = "class")
  
  #plot the decision tree
  #plot(tree_model, uniform = TRUE, compress = TRUE, margin = 0.1)
  #text(tree_model, use.n = TRUE, cex = 0.4) 
  
  test_data <- testing4(df)
  
  return(test_data)
}

#original dataframe
test_decision_tree_model <- decision_tree_model(df)
#Accuracy - 0.93 / Error rate - 0.07
#dataframe with the first 2 PCs and the cluster
test_decision_tree_model <- decision_tree_model(pcs_only)
#Accuracy - 0.93 / Error rate - 0.07
#dataframe with the original dataframe, the first 2 PCs and the cluster
test_decision_tree_model <- decision_tree_model(df_pcs_cluster)
#Accuracy - 0.93 / Error rate - 0.07
#the results lets try to add more pcs to the the dataframe with the first 2 PCs and the cluster
pcs_only <- data.frame()
pcs_only <- as.data.frame(pca$x[, 1:9])
pcs_only$status <- df$status
pcs_only$Cluster <- df_pcs_cluster$Cluster
test_decision_tree_model <- decision_tree_model(pcs_only)
#Accuracy - 0.93 / Error rate - 0.07


#TASK 4: KAGGLE COMPETETION--------------------------------------------------------------------------

#read the csv file
df_kaggle <- read_csv("kaggle.csv") 
print(colnames(df_kaggle))
View(df_kaggle)


#DATA PREPARATION

#we don't need "account_id", "date_diff" and  "status" (for now)
df_kaggle <- df_kaggle[, !colnames(df_kaggle) %in% c("account_id", "date_diff", "status")]
#normalize every numeric column (with z-score standardization), but "loan_id"
numeric_cols <- sapply(df_kaggle, is.numeric)
numeric_cols["loan_id"] <- FALSE
df_kaggle[numeric_cols] <- scale(df_kaggle[numeric_cols])
#label encode "type"
df_kaggle <- df_kaggle %>% mutate(type = recode(type, "credit" = 1, "withdrawal" = 2, "withdrawal in cash" = 3))

#PCA in relation to "loan_id"
pca_kaggle <- prcomp(df_kaggle, scale = TRUE)
summary(pca_kaggle)
#new dataframe with only  PCs
pcs_only_kaggle <- as.data.frame(pca_kaggle$x[, 1:7])
#add "loan_id" 
pcs_only_kaggle <- cbind(pcs_only_kaggle, loan_id = df_kaggle$loan_id)
view(pcs_only_kaggle)


#PREDICTIVE MODELLING

#using our previous decision trees model
tree_probabilities_kaggle <- predict(tree_model, pcs_only_kaggle[, names(pcs_only_kaggle) != "status_binary"], type = "prob")
#every probability
positive_class_probabilities <- tree_probabilities_kaggle[, 2] 
#add our predictions to the dataframe
pcs_only_kaggle <- cbind(pcs_only_kaggle, status = positive_class_probabilities)


#FINAL DATA MANIPULATION

#remove the PCs columns
pcs_only_kaggle <- pcs_only_kaggle[, !colnames(pcs_only_kaggle) %in% c("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7")]
#rename the columns "loan_id" to "Id" and "status" to "Predicted"
colnames(pcs_only_kaggle)[colnames(pcs_only_kaggle) == "loan_id"] <- "Id"
colnames(pcs_only_kaggle)[colnames(pcs_only_kaggle) == "status"] <- "Predicted"
view(pcs_only_kaggle)
#we only want 1 prediction for id, so we'll calculate the mean for each id
result <- pcs_only_kaggle %>%
  group_by(Id) %>%
  summarise(Predicted = mean(Predicted))
unique_ids <- unique(pcs_only_kaggle$Id)
final_result <- data.frame(Id = unique_ids, Predicted = sapply(unique_ids, function(id) mean(pcs_only_kaggle$Predicted[pcs_only_kaggle$Id == id])))
view(final_result)
#convert our final file to a CSV
write.csv(final_result, file = "decision_tree_result.csv", row.names = FALSE)

