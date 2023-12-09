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
library(pROC)

#restart R if necessary
#.rs.restartR()

#TASK 1.1: DATA UNDERSTANDING AND PREPARATION-----------------------------------------------------------

#DATABASE

#we're going to use the database with the original attributes for the descriptive modelling
#since we've more observations and we lose less information
df <- read_csv("new_db.csv") 
print(colnames(df))
#view(df)
#we don't need payments (its redundant)
df <- df[, !(names(df) %in% c("payments"))]

#DATA REDUCTION

#extract the p-values of the linear model
#for each attribute in relation with our target variable "status"
model <- lm(status ~ ., data = df)
p_values <- summary(model)$coefficients[, "Pr(>|t|)"]
formatted_p_values <- sprintf("%.5f", p_values)
result <- data.frame(attribute = names(p_values), p_values = as.numeric(formatted_p_values)
)
print(result)
#pvalue>0.05 -> not significant 

#calculate the correlations
#for each attribute in relation with our target variable "status"
correlations <- cor(df[, -which(names(df) == "status")], df$status)
print(correlations)
#correlation close to 0 -> not significant 

#we're not removing any attribute

#PCA
pca <- prcomp(df, scale = TRUE)
#summary(pca)
#visual representation of the PCA (1 and 2)
#biplot(pca, scale = 0, choices = c(1, 2))
#new dataframe with only 2 PCs
pcs_only <- as.data.frame(pca$x[, 1:2])


#TASK2: DESCRIPTIVE MODELLING-------------------------------------------------------------------------

#WITH THE DATAFRAME THAT ONLY HAS THE FIRST 2 PCS

#HIERARCHICAL CLUSTERING

#with euclidean and ward method
dist_matrix <- dist(pcs_only, method = "euclidean")
hclust_model <- hclust(dist_matrix, method = "ward.D2")
plot(hclust_model, hang = 0.1)
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
#we will choose 4 clusters, looking at the next plots, it's clear that we have
#4 different groups

#partitional clustering
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
#0.81

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
#0.79

#WITH THE ORIGINAL DATAFRAME AND THE FIRST 2 PCS

#creating our new dataframe
df_pcs <- df[, !colnames(df) %in% c("Cluster")]
pcs_only_2 <- pcs_only[, !colnames(pcs_only) %in% c("Cluster")]
df_pcs <- cbind(df_pcs, pcs_only_2)
#view(df_pcs)

#HIERARCHICAL CLUSTERING

#with euclidean and ward method
dist_matrix <- dist(df_pcs, method = "euclidean")
hclust_model <- hclust(dist_matrix, method = "ward.D2")
#plot(hclust_model, hang = 0.1)
#not readable

#PARTITIONAL CLUSTERING

#graph with the within cluster sum of squares to determine the ideal number of clusters
wcss <- vector()
for (i in 1:10) {
  kmeans_model <- kmeans(df_pcs, centers = i, nstart = 10)
  wcss[i] <- kmeans_model$tot.withinss
}
plot(1:10, wcss, type = "b", xlab = "Number of Clusters", ylab = "WCSS")
diff_wcss <- c(0, diff(wcss))
#we apply the elbow method, and the ideal number of clusters is 3 or 4
#we will choose 4 clusters, looking at the next plots, it's clear that we have
#4 different groups

#partitional clustering
num_clusters <- 4
cluster_cut <- cutree(hclust_model, k = num_clusters)
#adds a new column to the dataframe with the clusters id
df_pcs$Cluster <- as.factor(cluster_cut)

#visualize the clusters
ggplot(df_pcs, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point() +
  labs(title = "Cluster Analysis", x = "PC1", y = "PC2")

#silhouette coefficient to measure how good the partitional clustering is
sf <- silhouette(cluster_cut, dist(df_pcs))
sf_avg <- mean(sf[, 3])
print(sf_avg)
#0.51

#k-means (to numeric columns)
numeric_cols <- sapply(df_pcs, is.numeric)
df_pcs_numeric <- df_pcs[, numeric_cols]
kmeans_model <- kmeans(df_pcs, centers = num_clusters, nstart = 25)

#visualize the clusters from k-means
fviz_cluster(kmeans_model, data = df_pcs_numeric, geom = "point",
             ellipse.type = "t", ggtheme = theme_classic(),
             ellipse.alpha = 0.5, palette = "jco",
             main = "K-means Clustering Results",
             xlab = "PC1", ylab = "PC2")

#silhouette coefficient to measure how good the k-means is
sf_kmeans <- silhouette(kmeans_model$cluster, dist(df_pcs_numeric))
sf_kmeans_avg <- mean(sf_kmeans[,3])
print(sf_kmeans_avg) 
#0.50

#its very clear after comparing the plots and the silhouette coefficient results
#that clustering with only the PCs is way better than clustering with the original attributes plus 
#the PCs, because using only the PCs we're dealing with a much simpler dataset that holds most
#of the information, improving the interpretability for the clustering algorithms 

#TASK 1.2: DATA UNDERSTANDING AND PREPARATION-----------------------------------------------------------

#DATABASE

#we're going to use the database with the transformed attributes for the predictive modelling
#where we've just an observation per loan, since we want to predict if each loan is going to get paid 
df <- read_csv("new_db2.csv") 
print(colnames(df))
#view(df)

#DATA REDUCTION

#extract the p-values of the linear model
#for each attribute in relation with our target variable "status"
model <- lm(status ~ ., data = df)
p_values <- summary(model)$coefficients[, "Pr(>|t|)"]
formatted_p_values <- sprintf("%.5f", p_values)
result <- data.frame(attribute = names(p_values), formatted_p_values = as.numeric(formatted_p_values)
)
print(result)
#pvalue>0.05 -> not significant 

#calculate the correlations
#for each attribute in relation with our target variable "status"
correlations <- cor(df[, -which(names(df) == "status")], df$status)
print(correlations)
#correlation close to 0 -> not significant 

#"duration" is the only attribute with high p-value and correlation close to 0
#so we're going to remove it
df <- df[, !(names(df) %in% c("duration"))]

#PCA in relation to "loan_id"
pca <- prcomp(df, scale = TRUE)
summary(pca)
#visual representation of the PCA (1 and 2)
#biplot(pca, scale = 0, choices = c(1, 2))
#PCs that hold at least of the information
cumulative_variance <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
num_components <- which(cumulative_variance >= 0.8)[1]
print(num_components)
#6 PCs
#new dataframe with the PCS
pcs_only <- as.data.frame(pca$x[, 1:num_components])

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
#view(df)
split_df <- split_data(df)
train_data_split_df <- split_df$train_data
test_data_split_df <- split_df$test_data

#split the data of the dataframe with the PCS
pcs_only <- pcs_only %>% mutate(loan_id = df$loan_id)
pcs_only <- pcs_only %>% mutate(status = df$status)
#view(pcs_only)
split_pcs_only <- split_data(pcs_only)
train_data_split_pcs_only <- split_pcs_only$train_data
test_data_split_pcs_only <- split_pcs_only$test_data

#split the data of the dataframe with the original dataframe and the PCS 
pcs_only <- pcs_only[, !(names(pcs_only) %in% c("status", "loan_id"))]
dfs_pcs <- cbind(df, pcs_only)
#view(dfs_pcs)
split_dfs_pcs <- split_data(dfs_pcs)
train_data_split_dfs_pcs <- split_dfs_pcs$train_data
test_data_split_dfs_pcs <- split_dfs_pcs$test_data
pcs_only <- pcs_only %>% mutate(loan_id = df$loan_id)
pcs_only <- pcs_only %>% mutate(status = df$status)

#NAIVE BAYES

naive_bayes_model <- function(train_data, test_data) {
  #training
  train_data$status <- as.factor(train_data$status)
  naive_model <- naive_bayes(train_data[, -which(names(train_data) == "status")], train_data$status, laplace = 1)
  
  #testing
  test_data$status <- as.factor(test_data$status)
  test_data$predicted <- predict(naive_model, newdata = test_data[, -which(names(test_data) == "status")])
  
  #calculating accuracy
  correct_predictions <- sum(test_data$predicted == test_data$status)
  total_predictions <- length(test_data$status)
  accuracy <- correct_predictions / total_predictions
  print(paste("Accuracy:", accuracy))
  
  #calculating error rate
  error_rate <- 1 - accuracy
  print(paste("Error rate:", error_rate))
  
  #calculating AUC
  roc_nb <- pROC::roc(test_data$status, as.numeric(test_data$predicted))
  print(paste("AUC:", pROC::auc(roc_nb)))
  
  return(test_data)
}

#original dataframe
test_naive_bayes_model <- naive_bayes_model(train_data_split_df, test_data_split_df)
#Accuracy - 0.39 / Error rate - 0.61 / AUC - 0.64
#dataframe with the PCS
test_naive_bayes_model <- naive_bayes_model(train_data_split_pcs_only, test_data_split_pcs_only)
#Accuracy - 0.98 / Error rate - 0.02 / AUC - 0.95
#dataframe with the original dataframe and  the PCS
test_naive_bayes_model <- naive_bayes_model(train_data_split_dfs_pcs, test_data_split_dfs_pcs)
#Accuracy - 0.79 / Error rate - 0.21 / AUC - 0.88

#LOGISTIC REGRESSION


#DECISION TREES

decision_tree_model <- function(df){
  #training
  tree_model <- rpart(df$status ~ ., data = df[, !(names(df) %in% c("status"))], method = "class")
  
  #testing
  predicted <- predict(tree_model, type = "class")
  
  #plot the decision tree
  #plot(tree_model, uniform = TRUE, compress = TRUE, margin = 0.1)
  #text(tree_model, use.n = TRUE, cex = 0.4) 
  
  #calculating accuracy
  accuracy <- mean(predicted == df$status)
  print(paste("Accuracy:", accuracy))
  
  #calculating error rate
  error_rate <- 1 - accuracy
  print(paste("Error rate:", error_rate))
  
  #calculating AUC
  prob <- predict(tree_model, type = "prob")
  auc_tree <- roc(df$status, prob[,2])
  print(paste("AUC:", auc(auc_tree)))
  
  return(test_data)
}

#original dataframe
test_decision_tree_model <- decision_tree_model(df)
#Accuracy - 0.91 / Error rate - 0.0.09 / AUC - 0.85
#dataframe with the PCS 
test_decision_tree_model <- decision_tree_model(pcs_only)
#Accuracy - 0.99 / Error rate -  0.01 / AUC - 0.98
#dataframe with the original dataframe and the PCS 
test_decision_tree_model <- decision_tree_model(df_pcs_cluster)
#Accuracy - 0.99 / Error rate - 0.01 / AUC - 0.99

#KNN

knn_model <- function(train_data, test_data, k) {
  #split the predictors and the target variable
  predictors_train <- train_data[, !(names(train_data) %in% c("status"))]
  target_train <- train_data$status
  predictors_test <- test_data[, !(names(test_data) %in% c("status"))]
  target_test <- test_data$status
  
  #training
  knn_predicted <- knn(train = predictors_train, test = predictors_test, cl = target_train, k = k)
  
  #calculating accuracy
  accuracy <- mean(knn_predicted == target_test)
  print(paste("Accuracy:", accuracy))
  
  #calculating error rate
  error_rate <- 1 - accuracy
  print(paste("Error rate:", error_rate))
  
  #calculating AUC
  knn_prob <- as.numeric(knn(train = predictors_train, test = predictors_test, cl = target_train, prob = TRUE, k = k))
  auc_knn <- roc(target_test, knn_prob)
  print(paste("AUC:", auc(auc_knn)))
  
  return(test_data)
}

#we determined the ideal number of ks is 5, after testing every k from 1 to 6
#original dataframe
results_knn <- knn_model(train_data_split_df, test_data_split_df, 5)
#Accuracy -  0.83 / Error rate - 0.17 / AUC - 0.49
#dataframe with the first 2 PCs and the cluster
results_knn <- knn_model(train_data_split_pcs_only, test_data_split_pcs_only, 5)
#Accuracy - 0.82 / Error rate - 0.18 / AUC - 0.48
#dataframe with the original dataframe and the PCS 
results_knn <- knn_model(train_data_split_dfs_pcs, test_data_split_dfs_pcs, 5)
#Accuracy - 0.83 / Error rate - 0.17 / AUC - 0.49