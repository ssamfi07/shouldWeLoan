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

#read the csv files
df <- read_csv("database1.csv") 
print(colnames(df))

#restart R if necessary
#.rs.restartR()

#TASK1: DATA UNDERSTANDING AND PREPARATION------------------------------------------------------------

#DATA CLEANING

#we don't need "loan_id", "trans_id"
df <- df[, !colnames(df) %in% c("loan_id", "trans_id")]
#convert data_diff to numeric
df$date_diff <- as.numeric(df$date_diff) 

#check for missing values
colSums(is.na(df))
#we have missing values in the columns "date_diff"
#so we need to change the N/A values to "0"
df$date_diff[is.na(df$date_diff)] <- 0
colSums(is.na(df))

#check for redundant or irrelevant values (or columns)
#check for columns with 0 variance
numeric_df <- df[sapply(df, is.numeric)]
zero_variance_cols <- colnames(numeric_df)[apply(numeric_df, 2, function(x) length(unique(x))) == 1]
print(zero_variance_cols)
#no column has 0 variance

#DATA TRANSFORMATION

#normalize every numeric column (with z-score standardization), but the target variable "status"
numeric_cols <- sapply(df, is.numeric)
numeric_cols["status"] <- FALSE
df[numeric_cols] <- scale(df[numeric_cols])

#check for categorical columns
categorical_cols <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
print(categorical_cols)
#only the column "type" is categorical
#so we are going to label encode it
df <- df %>% mutate(type = recode(type, "credit" = 1, "withdrawal" = 2, "withdrawal in cash" = 3))


#DATA REDUCTION

#(Pearson) correlation in relation to "status" (our target variable)
correlations <- sapply(df, function(col) cor(col, df$status))
print(correlations)
#the columns that have almost no correlation with "status" will be dropped 
#only "date_diff" and "amount_trans" will be dropped
df <- df[, !colnames(df) %in% c("amount_trans", "date_diff")]

#PCA in relation to "account_id"
pca <- prcomp(df, scale = TRUE)
#visual representation of the PCA (1 and 2)
biplot(pca, scale = 0, choices = c(1, 2))
#new dataframe with only 2 PCs
pcs_only <- as.data.frame(pca$x[, 1:2])
#new dataframe with the original columns and the first 2 PCs
df_with_pcs <- cbind(df, pca$x[, 1:2])


#TASK2: DESCRIPTIVE MODELLING-------------------------------------------------------------------------

#HIERARCHICAL CLUSTERING

#with euclidean and ward method
dist_matrix <- dist(df, method = "euclidean")
hclust_model <- hclust(dist_matrix, method = "ward.D2")
plot(hclust_model, hang = 0.1)
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
num_clusters <- 3
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
#0.30 is not a very good result

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
#3 clusters seems to be the ideal number of clusters

#silhouette coefficient to measure how good the k-means is
sf_kmeans <- silhouette(kmeans_model$cluster, dist(pcs_only_numeric))
sf_kmeans_avg <- mean(sf_kmeans[,3])
print(sf_kmeans_avg) 
#0.36 is not a very good result


#TASK 3: PREDICTIVE MODELLING-------------------------------------------------------------------------

#SPLIT THE DATA

#random split
set.seed(123)
#80% training and 20% testing
train_indices <- sample(1:nrow(df), 0.8*nrow(df))
#training dataframe
train_data <- df[train_indices, ]
#testing dataframe
test_data <- df[-train_indices, ]


#NAIVE BAYES

#training all columns except our target variable "status" (we need to convert it to a factor)
train_data$status <- as.factor(train_data$status)
naive_model <- naive_bayes(train_data[, -which(names(train_data) == "status")], train_data$status)

#testing  all columns except our target variable "status"
test_data$predicted <- predict(naive_model, newdata = test_data[, -which(names(test_data) == "status")])

#calculating accuracy
correct_predictions <- sum(test_data$predicted == test_data$status)
total_predictions <- length(test_data$status)
accuracy <- correct_predictions / total_predictions
print(paste("Accuracy:", accuracy))
#0.86

#calculating error rate
error_rate <- 1 - accuracy
print(paste("Error rate:", error_rate))
#0.14


#KNN


#LOGISTIC REGRESSION

#convert our target variable to binary
df$status_binary <- ifelse(df$status == 1, 1, 0)
#apply our model
logistic_model <- glm(df$status_binary ~ ., data = df[, !(names(df) %in% c("status", "status_binary"))], family = "binomial")
#make predictions
df$predicted <- predict(logistic_model, type = "response")


#calculating accuracy
df$predicted_binary <- ifelse(df$predicted >= 0.5, 1, 0)
accuracy <- mean(df$predicted_binary == df$status_binary)
print(paste("Accuracy:", accuracy))
#0.88

#calculating error rate
error_rate <- 1 - accuracy
print(paste("Error rate:", error_rate))
#0.12


#DECISION TREES

#apply our model
tree_model <- rpart(df$status_binary ~ ., data = df[, !(names(df) %in% c("status", "status_binary"))], method = "class")

#calculating accuracy
predicted <- predict(tree_model, type = "class")
accuracy <- mean(predicted == df$status_binary)
cat("Accuracy:", accuracy, "\n")
#0.95

#calculating error rate
error_rate <- 1 - accuracy
cat("Error rate:", error_rate, "\n")
#0.05

#plot the decision tree
plot(tree_model, uniform = TRUE, compress = TRUE, margin = 0.1)
text(tree_model, use.n = TRUE, cex = 0.4) 