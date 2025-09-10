

```{r}
# GAM improvements 

gam.check(gam_model_1) # 
summary(gam_model_1)

# suspect high corr between longitude and latitude 
# should add interaction term 
plot(data$longitude, data$latitude)

# same thing with aveRooms and aveBedrooms 
plot(data$aveRooms, data$aveBedrooms)
```








## LDA is actually not as good as improved KNN

```{r}
# LDA 

lda_fit <- lda(censoring ~ longitude + 
                 latitude + 
                 housingMedianAge + 
                 aveRooms + 
                 aveBedrooms + 
                 population + 
                 medianIncome + 
                 oceanProximity, 
               data = train_data_c)

lda_pred <- predict(lda_fit, test_data_c)$class 

#library(caret)
confusionMatrix(as.factor(lda_pred), as.factor(test_data_c$censoring))

(lda_err = mean(lda_pred != test_data_c$censoring)) 
cat("The LDA error rate is", round(lda_err, 4), "\n")
# 0.0343 
```



```{r}
# LDA with improvements

lda_fit <- lda(censoring ~ longitude + 
                 latitude + 
                 medianIncome + 
                 housingMedianAge + 
                 oceanProximity + 
                 aveRooms + 
                 aveBedrooms +
                 cityProximityScore + 
                 bedroomsPerRoom + 
                 incomePerRoom +
                 distToSF +
                 distToLA +
                 cosDirToLA +
                 cosDirToSF +
                 sinDirToLA +
                 sinDirToSF, 
               data = train_data_c_imp)

lda_pred <- predict(lda_fit, test_data_c_imp)$class 

#library(caret)
confusionMatrix(as.factor(lda_pred), as.factor(test_data_c_imp$censoring))

(lda_err = mean(lda_pred != test_data_c_imp$censoring)) 
cat("The LDA error rate is", round(lda_err, 4), "\n")
#0.0345 
# so worse
```
















\noindent \large \textbf{Support Vector Machine (SVM)} \normalsize

\begin{enumerate} 
\item \textbf{Effevtive for high-dimensionality}: Since we have a large amount of features, SVM are suitable due to its kernel approach. 
\item \textbf{Avoids ovefitting by maximising margin}: SVM focuses on maximizing the margin between the classes (i.e. it tries to be as far away as possible from the closest points in each class (called support vectors)). This reduces the model's sensitivity to small changes in data, which helps the model perform better on unseen data.
\item \textbf{Accomodating non-linear relationships}: SVMs can be extended to handle complex decision boundaries through the kernel appraoch, allowing modeling of nonlinear relationships.

\noindent \large \textbf{Naïve Bayes} \normalsize






## SVM is very slow, idk if it really works tbh

```{r}
# Load required package
library(e1071)
library(caret)

# Split data (in case it's not already done)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Define formula
svm_formula <- censoring ~ longitude + latitude + medianIncome + housingMedianAge + 
  oceanProximity + aveRooms + aveBedrooms

#-------------------------------
# Tuning and fitting models
#-------------------------------

# LINEAR KERNEL
set.seed(5)
tune_linear <- tune(svm, svm_formula, data = train_data, 
                    kernel = "linear", 
                    ranges = list(cost = c(0.01, 0.1, 1, 10, 100)),
                    scale = TRUE)
svm_linear <- tune_linear$best.model

# RADIAL KERNEL
set.seed(5)
tune_radial <- tune(svm, svm_formula, data = train_data, 
                    kernel = "radial", 
                    ranges = list(cost = c(0.01, 0.1, 1, 10, 100)),
                    scale = TRUE)
svm_radial <- tune_radial$best.model

# POLYNOMIAL KERNEL
set.seed(5)
tune_poly <- tune(svm, svm_formula, data = train_data, 
                  kernel = "polynomial", degree = 2,
                  ranges = list(cost = c(0.01, 0.1, 1, 10, 100)),
                  scale = TRUE)
svm_poly <- tune_poly$best.model

#-------------------------------
# Prediction and Evaluation
#-------------------------------

# Function to evaluate and print confusion matrix and error rate
evaluate_svm <- function(model, name) {
  pred <- predict(model, test_data)
  true <- factor(test_data$censoring)
  pred <- factor(pred, levels = levels(true))  # align factor levels
  cm <- confusionMatrix(pred, true)
  print(cm)
  err <- mean(pred != true)
  cat(name, "SVM test error rate:", round(err, 4), "\n\n")
}

# Evaluate each model
evaluate_svm(svm_linear, "Linear")
evaluate_svm(svm_radial, "Radial")
evaluate_svm(svm_poly,   "Polynomial")




```




```{r}
# SVM 

# install.packages("e1071")
library(e1071)
train_data <- data[train_index, ] 
# in case any modification of the data set needed gotta rerun this 
test_data <- data[-train_index, ]

svm_model_1 <- svm(censoring ~ longitude + latitude + medianIncome + housingMedianAge + 
                     oceanProximity + aveRooms + aveBedrooms, 
                   data = train_data, 
                   kernel = "linear",  # try radial later 
                   cost = 1,          # Regularization parameter (tune this)
                   scale = TRUE)      # Standardize the predictors

svm_model_2 <- svm(censoring ~ longitude + latitude + medianIncome + housingMedianAge + 
                     oceanProximity + aveRooms + aveBedrooms, 
                   data = train_data, 
                   kernel = "radial",  
                   cost = 1,        # Regularization parameter (tune this)
                   scale = TRUE)  

svm_model_2 <- svm(censoring ~ longitude + latitude + medianIncome + housingMedianAge + 
                     oceanProximity + aveRooms + aveBedrooms, 
                   data = train_data, 
                   kernel = "polynomial",  
                   cost = 1,        # Regularization parameter (tune this)
                   scale = TRUE) 
svm_fits <- list(
  "Radial" = svm(censoring ~ 
                   longitude + latitude + medianIncome + 
                   housingMedianAge +  aveRooms + aveBedrooms, 
                 data = data[train_index, ],
                 kernel = "radial"), 
  "Polynomial" = svm(censoring ~ 
                       longitude + latitude + medianIncome + 
                       housingMedianAge +  aveRooms + aveBedrooms, 
                     data = data[train_index, ],
                     kernel = "polynomial", degree = 2), 
  "Linear" = svm(censoring ~ 
                   longitude + latitude + medianIncome + 
                   housingMedianAge + aveRooms + aveBedrooms, 
                 data = data[train_index, ],
                 kernel = "linear"))

err <- function(model, data) {
  out <- table(predict(model, data), data$censoring)
  (out[1, 2] + out[2, 1]) / sum(out)
}

plot(svm_fits[[1]], datap[train_index, ])




# Predict on the test set
svm_pred_linear <- predict(svm_fits[["Linear"]], test_data)
svm_pred_radial <- predict(svm_fits[["Radial"]], test_data)
svm_pred_poly   <- predict(svm_fits[["Polynomial"]], test_data)

# Compute test error and confusion matrix
confusionMatrix(as.factor(svm_pred_linear), as.factor(test_data$censoring))
(svm_err_linear <- mean(svm_pred_linear != test_data$censoring))
cat("Linear SVM error rate:", round(svm_err_linear, 4), "\n")

confusionMatrix(as.factor(svm_pred_radial), as.factor(test_data$censoring))
(svm_err_radial <- mean(svm_pred_radial != test_data$censoring))
cat("Radial SVM error rate:", round(svm_err_radial, 4), "\n")

confusionMatrix(as.factor(svm_pred_poly), as.factor(test_data$censoring))
(svm_err_poly <- mean(svm_pred_poly != test_data$censoring))
cat("Polynomial SVM error rate:", round(svm_err_poly, 4), "\n")


```


```{r}
```


```{r}
# Naive Bayes 

nb_fit <- naiveBayes(censoring ~ longitude + 
                       latitude + 
                       housingMedianAge + 
                       aveRooms + 
                       aveBedrooms + 
                       population + 
                       medianIncome, 
                     data = data[train_index, ])

nb_pred <- predict(nb_fit, data[-train_index, ], type = "class")

# Confusion table 
(nb_t <- table(nb_pred, data[-train_index, ]$censoring))

(nb_error = 1 - sum(diag(nb_t)) / sum(nb_t))
cat("The Naive Bayes error rate is", round(nb_error, 4), "\n")

# not doing as good as logistic 
# 0.0507 error rate

```
```{r}
# Alina code

test_data <- data[-train_index, ]
#QDA
#fit qda model
qda_fit <- qda(censoring ~ 
                 longitude + latitude + medianIncome + 
                 housingMedianAge + aveRooms + aveBedrooms, 
               data = data[train_index, ])

#Predict on test data
qda_pred <- predict(qda_fit, data[-train_index,])$class

# Confusion matrix
confusionMatrix(as.factor(qda_pred), as.factor(test_data$censoring))

#error rate
qda_err <- mean(qda_pred != test_data$censoring ) 
cat("The QDA error rate is", round(qda_err, 4), "\n")
# error rate: 0.0426 
```











```{r}
# Alina code

#wip but 2.2.4 potentially?

#decision boundary visualisation

# Compute posterior probs for positive class (censoring == 1)

grid$lda_post <- predict(lda_fit, grid)$posterior[, "1"]
grid$qda_post <- predict(qda_fit, grid)$posterior[, "1"]
grid$nb_post  <- predict(nb_fit, grid, type = "raw")[, "1"]

library(ggplot2)

ggplot(data, aes(x = longitude, y = latitude)) +
  geom_point(aes(color = as.factor(censoring)), alpha = 0.5) +

  # Decision boundaries at 0.5 posterior probability - not showing at 0.5
  geom_contour(data = grid, aes(z = lda_post), breaks = 0.005,
               color = "black", linetype = "dotted") +

  geom_contour(data = grid, aes(z = qda_post), breaks = 0.006,
               color = "green", linetype = "solid") +

  geom_contour(data = grid, aes(z = nb_post), breaks = 0.0001,
               color = "purple", linetype = "dashed") +

  labs(title = "Decision Boundaries: LDA (black), QDA (green), NB/Bayes (purple)",
       color = "Censoring") +
  theme_minimal()

#boxplot - compare methods
library(MASS)
library(e1071)
library(caret)
library(ggplot2)

set.seed(1)

n_reps <- 30
test_errors <- data.frame(Method = character(), Error = numeric())

for (i in 1:n_reps) {
  train_idx <- createDataPartition(data$censoring, p = 0.8, list = FALSE)
  train <- data[train_idx, ]
  test <- data[-train_idx, ]
  
  if (length(unique(train$censoring)) < 2) next
  
  # Fit LDA
  lda_fit <- lda(censoring ~ longitude + latitude + medianIncome + housingMedianAge +
                 aveRooms + aveBedrooms + population + oceanProximity, data = train)
  lda_pred <- predict(lda_fit, test)$class
  lda_err <- mean(lda_pred != test$censoring)
  
  # Fit QDA with error handling
  qda_err <- NA
  qda_fit <- tryCatch({
    qda(censoring ~ longitude + latitude + medianIncome + housingMedianAge +
        aveRooms + aveBedrooms + population + oceanProximity, data = train)
  }, error = function(e) NULL)
  
  if (!is.null(qda_fit)) {
    qda_pred <- predict(qda_fit, test)$class
    qda_err <- mean(qda_pred != test$censoring)
  } else {
    message(sprintf("QDA failed on iteration %d — skipping", i))
  }
  
  # Fit nb
  nb_fit <- naiveBayes(censoring ~ longitude + latitude + medianIncome + housingMedianAge +
                       aveRooms + aveBedrooms + population + oceanProximity, data = train)
  nb_pred <- predict(nb_fit, test)
  nb_err <- mean(nb_pred != test$censoring)
  
  # Store results only if QDA worked; not sure if bc too many predictors?
  if (!is.null(qda_fit)) {
    test_errors <- rbind(test_errors,
                         data.frame(Method = "LDA", Error = lda_err),
                         data.frame(Method = "QDA", Error = qda_err),
                         data.frame(Method = "Naive Bayes", Error = nb_err))
  } else {
    # Store only LDA and NB errors
    test_errors <- rbind(test_errors,
                         data.frame(Method = "LDA", Error = lda_err),
                         data.frame(Method = "Naive Bayes", Error = nb_err))
  }
}

ggplot(test_errors, aes(x = Method, y = Error, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Test Error Comparison with QDA Error Handling",
       y = "Test Error",
       x = "Method") +
  theme_minimal() +
  theme(legend.position = "none")
```
