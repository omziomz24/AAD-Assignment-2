

### A probability-weighted approach 

The idea behind the probability-weighted approach is to treat the final house price prediction as the expected value under uncertainty about whether a value is censored. 

We don’t know for sure whether an observation is censored, but we have a probability estimate from the logistic model. So, instead of making a hard decision, we compute the weighted average of:
  
  the threshold value ($500{,}001$) if it is censored, and
the KNN regression prediction if it is not censored,
weighted by the probability of each case.

Let \( p_i = \mathbb{P}(\text{censored} \mid \text{features}_i) \), and let \( \hat{y}_i^{\text{KNN}} \) be the KNN regression prediction. The probability-weighted hybrid prediction is then:
  
  \[
    \hat{y}^*_i = p_i \cdot 500001 + (1 - p_i) \cdot \hat{y}_i^{\text{KNN}}
    \]

This is interpreted as the expected value of the house price under uncertainty about censoring.

This adjusted hybrid prediction approach is highly feasible, as it simply disregard the 0.5 threshold in the hard-decision classification previously, and simply use the probability output from logistic regression in the medianHousingValue prediction. This eliminates the difficulty about choosing an appropriate threshold, as well as avoid omission of useful information due to a fixed, arbitrary cutoff (e.g. \ 0.5). 

```{r}
# Some adjustments to the hard decision classification? 

# Step 1: Predict house prices with KNN regression (same as before)
knn_pred_2 <- predict(knn_reg_model_2, newdata = test_data_encoded)

# Step2: Predict censoring with logistic regression
# Instead of the hard classification decision, use 
logit_prob_2 <- predict(logit_fit_2, newdata = test_data_c_imp, type = "response")

# Step 3: Adjust predicted house prices
adjusted_preds_ver2 <- logit_prob_2 * 500001 + (1 - logit_prob_2) * knn_pred_2
# this is like a expected value 

# Step4: MSE calculation
final_mse_ver2 <- mean((adjusted_preds_ver2 - test_data_encoded$medianHouseValue)^2)

(final_mse_ver2) 

# final_mse_ver2 is 2975741318

cat("Final Test MSE using probability-weighted prediction:", round(final_mse_ver2, 2), "\n")
```

The modified hybrid model has a slightly lower MSE (2,975,741,318) compared to the KNN regression model from 2.1.3 (2,999,761,722). This improvement can be attributed to how the probability-weighted method handles uncertainty near the censoring boundary. Instead of applying a hard classification rule (i.e., clipping to 500,001 when the probability of censoring exceeds 0.5), this method uses a soft probabilistic adjustment. This approach acts like an expected value under uncertainty, allowing smoother transitions near the threshold.

However, this 0.8% decrease is not significant. It may fall within the range of random variation due to the specific train-test split or model variability. While the probabilistic adjustment offers a more refined treatment of censoring, the overall improvement is marginal, suggesting that both approaches perform similarly in practice.

















#### What if we apply outer-cv to compare: 

```{r}
# library(caret)
# library(dplyr)
```


```{r}
# define data_encoded 

library(recipes)

# Build a recipe to handle missing values and encoding
housing_recipe <- recipe(medianHouseValue ~ ., data = data) %>%
  step_impute_median(all_numeric_predictors()) %>%  # updated function
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_predictors())  # scales 0–1 for KNN

# Prep and bake
housing_prep <- prep(housing_recipe)
data_encoded <- bake(housing_prep, new_data = NULL)
```

```{r}
library(caret)
set.seed(123)

# Define tuning grid for k from 1 to 15
tune_grid <- expand.grid(k = 1:15)

# Train KNN regression with 5-fold CV
knn_fit <- train(
  medianHouseValue ~ .,
  data = train_data_encoded,
  method = "knn",
  tuneGrid = tune_grid,
  trControl = trainControl(method = "cv", number = 5)
)

# Best k
knn_fit$bestTune

# Plot CV results
plot(knn_fit)

# Predict on test data
knn_preds <- predict(knn_fit, newdata = test_data_encoded)

# Calculate MSE
mse <- mean((knn_preds - test_data_encoded$medianHouseValue)^2)
cat("Test MSE with tuned KNN:", round(mse, 2), "\n")
```
