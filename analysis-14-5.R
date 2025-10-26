library(lubridate)
library(ggplot2)
library(gridExtra)
library(moments)
library(corrplot)
library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(keras)
library(car)
library(dplyr)
library(tibble)
library(dplyr)
library(purrr)


set.seed(123)

if (requireNamespace("here", quietly = TRUE)) {
  data <- read.csv(here::here("ev_charging_patterns.csv"))
} else {
  data <- read.csv("ev_charging_patterns.csv")
}

# ensure figures folder exists for saving plots
fig_dir <- "figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
#Explanatory data analysis

#Fix time of day

# Ensure Charging.Start.Time is in datetime format
data$Charging.Start.Time <- ymd_hms(data$Charging.Start.Time)

# Define function to classify time of day
get_time_of_day <- function(hour) {
  if (hour >= 5 & hour < 12) {
    return('Morning')
  } else if (hour >= 12 & hour < 17) {
    return('Afternoon')
  } else if (hour >= 17 & hour < 21) {
    return('Evening')
  } else {
    return('Night')
  }
}

# Apply the function using the correct data frame and column
data$CorrectedTimeOfDay <- sapply(hour(data$Charging.Start.Time), get_time_of_day)

# View the results
table(data$CorrectedTimeOfDay)

head(data[, c("Charging.Start.Time", "CorrectedTimeOfDay")], 20)


# Create a new dataframe with the relevant explanatory variables and the target var
df_analysis <- data.frame(
  Energy.Consumed.kWh              = data$Energy.Consumed..kWh.,
  Battery.Capacity.kWh             = data$Battery.Capacity..kWh.,
  Charging.Rate.kW                 = data$Charging.Rate..kW.,
  Charging.Cost.USD                = data$Charging.Cost..USD.,
  CorrectedTimeOfDay               = data$CorrectedTimeOfDay,
  Day.of.Week                      = data$Day.of.Week,
  State.of.Charge.Start            = data$State.of.Charge..Start...,
  Distance.Since.Last.Charge.km    = data$Distance.Driven..since.last.charge...km.,
  Temperature.C                   = data$Temperature...C.,
  Vehicle.Age.years                = data$Vehicle.Age..years.,
  Charger.Type                     = data$Charger.Type,
  User.Type                        = data$User.Type
)

names(df_analysis)[1] <- "energy_kwh"

df_analysis <- subset(
  df_analysis,
  energy_kwh <= Battery.Capacity.kWh &
  State.of.Charge.Start <= 100
)

# quick sanity checks
stopifnot(all(df_analysis$energy_kwh            <= df_analysis$Battery.Capacity.kWh))
stopifnot(all(df_analysis$State.of.Charge.Start <= 100))

str(df_analysis)
summary(df_analysis)

#Transformation of characters to factors
df_analysis$CorrectedTimeOfDay <- as.factor(df_analysis$CorrectedTimeOfDay)
df_analysis$Day.of.Week <- as.factor(df_analysis$Day.of.Week)
df_analysis$Charger.Type <- as.factor(df_analysis$Charger.Type)
df_analysis$User.Type <- as.factor(df_analysis$User.Type)

str(df_analysis)

#check levels
levels(df_analysis$CorrectedTimeOfDay)
levels(df_analysis$Day.of.Week)
levels(df_analysis$Charger.Type)
levels(df_analysis$User.Type)

#handle missing values
which_rows_na <- apply(df_analysis, 1, function(x) any(is.na(x)))
sum(which_rows_na)  # total rows with ANY NA

df_analysis_na <- df_analysis[which_rows_na, ]
summary(df_analysis_na)

#drop the missing values
df_analysis <- na.omit(df_analysis)
#Confirmation checks
colSums(is.na(df_analysis))  # should return all zeros
nrow(df_analysis)            # confirm reduced from 1320 to 1131

# Outlier Detection

# Create boxplots to visualize outliers
par(mfrow=c(2,3))
boxplot(df_analysis$energy_kwh, main="Energy Consumed (kWh)", col="lightblue")
boxplot(df_analysis$Battery.Capacity.kWh, main="Battery Capacity (kWh)", col="lightgreen")
boxplot(df_analysis$Charging.Rate.kW, main="Charging Rate (kW)", col="lightpink")
boxplot(df_analysis$Charging.Cost.USD, main="Charging Cost (USD)", col="lightyellow")
boxplot(df_analysis$State.of.Charge.Start, main="Initial State of Charge (%)", col="lightcyan")
boxplot(df_analysis$Distance.Since.Last.Charge.km, main="Distance Since Last Charge (km)", col="coral")
par(mfrow=c(1,1))

# Identify outliers using IQR method
identify_outliers <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  return(which(x < lower_bound | x > upper_bound))
}

# Apply to key variables
energy_outliers <- identify_outliers(df_analysis$energy_kwh)
battery_outliers <- identify_outliers(df_analysis$Battery.Capacity.kWh)
rate_outliers <- identify_outliers(df_analysis$Charging.Rate.kW)
cost_outliers <- identify_outliers(df_analysis$Charging.Cost.USD)

# Create a subset to examine the outliers
all_outliers <- unique(c(energy_outliers, battery_outliers, rate_outliers, cost_outliers))
outliers_df <- df_analysis[all_outliers, ]

# Energy consumption distribution
p1 <- ggplot(df_analysis, aes(x=energy_kwh)) + 
  geom_histogram(bins=30, fill="steelblue", alpha=0.7) +
  labs(title="Distribution of Energy Consumed", x="Energy (kWh)", y="Frequency")

# Energy consumption by user type
p2 <- ggplot(df_analysis, aes(x=User.Type, y=energy_kwh, fill=User.Type)) + 
  geom_boxplot() +
  labs(title="Energy Consumption by User Type", x="User Type", y="Energy (kWh)") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Energy consumption by time of day
p3 <- ggplot(df_analysis, aes(x=CorrectedTimeOfDay, y=energy_kwh, fill=CorrectedTimeOfDay)) + 
  geom_boxplot() +
  labs(title="Energy Consumption by Time of Day", x="Time of Day", y="Energy (kWh)")

# Energy consumption by day of week
p4 <- ggplot(df_analysis, aes(x=Day.of.Week, y=energy_kwh, fill=Day.of.Week)) + 
  geom_boxplot() +
  labs(title="Energy Consumption by Day of Week", x="Day of Week", y="Energy (kWh)") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Arrange plots
grid.arrange(p1, p2, p3, p4, ncol=2)

# Charger type usage patterns
ggplot(df_analysis, aes(x=Charger.Type, fill=User.Type)) + 
  geom_bar(position="dodge") +
  labs(title="Charger Type Usage by User Type", x="Charger Type", y="Count") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Charging behavior by temperature
ggplot(df_analysis, aes(x=Temperature.C, y=energy_kwh, color=User.Type)) + 
  geom_point(alpha=0.6) +
  geom_smooth(method="loess") +
  labs(title="Energy Consumption vs Temperature", x="Temperature (°C)", y="Energy (kWh)")

# Correlation Analysis

# Create a subset of numeric variables for correlation analysis
numeric_vars <- df_analysis[, sapply(df_analysis, is.numeric)]

# Correlation matrix
cor_matrix <- cor(numeric_vars, use="complete.obs")
print(cor_matrix)

corrplot(cor_matrix, method="circle", type="upper", 
         tl.col="black", tl.srt=45, addCoef.col="black", 
         number.cex=0.7)

# Categorical variable association (Chi-square test)
chisq.test(table(df_analysis$User.Type, df_analysis$Charger.Type))
chisq.test(table(df_analysis$CorrectedTimeOfDay, df_analysis$User.Type))

# Analysis of variance (ANOVA) to test mean differences across groups
anova_user_type <- aov(energy_kwh ~ User.Type, data=df_analysis)
summary(anova_user_type)

anova_time_of_day <- aov(energy_kwh ~ CorrectedTimeOfDay, data=df_analysis)
summary(anova_time_of_day)

anova_charger_type <- aov(energy_kwh ~ Charger.Type, data=df_analysis)
summary(anova_charger_type)

# Scatter plot matrix for key variables
pairs(~ energy_kwh + Battery.Capacity.kWh + Charging.Rate.kW + State.of.Charge.Start + 
        Distance.Since.Last.Charge.km + Temperature.C, data=df_analysis,
      main="Scatter Plot Matrix", pch=19, col=rgb(0,0,1,0.3))

# Advanced Analysis for Key Relationships

# Interaction between temperature and charging behavior
ggplot(df_analysis, aes(x=Temperature.C, y=energy_kwh, color=Charger.Type)) +
  geom_point(alpha=0.6) +
  geom_smooth(method="lm") +
  facet_wrap(~User.Type) +
  labs(title="Temperature Effect on Charging by User and Charger Type", 
       x="Temperature (°C)", y="Energy Consumed (kWh)")

# Relationship between state of charge and energy consumed
ggplot(df_analysis, aes(x=State.of.Charge.Start, y=energy_kwh, color=Battery.Capacity.kWh)) +
  geom_point() +
  scale_color_viridis_c() +
  geom_smooth(method="lm", color="red") +
  labs(title="Initial State of Charge vs Energy Consumed", 
       x="Initial State of Charge (%)", y="Energy Consumed (kWh)")

# Analysis of user behavior patterns
ggplot(df_analysis, aes(x=Distance.Since.Last.Charge.km, y=State.of.Charge.Start, color=User.Type)) +
  geom_point(alpha=0.7) +
  geom_smooth(method="loess") +
  labs(title="Charging Behavior Patterns by User Type", 
       x="Distance Since Last Charge (km)", y="Initial State of Charge (%)")


# Aggregate statistics by user type
user_type_stats <- aggregate(energy_kwh ~ User.Type, data=df_analysis, 
                             FUN=function(x) c(mean=mean(x), median=median(x), sd=sd(x)))
print(user_type_stats)

# Aggregate statistics by time of day
time_of_day_stats <- aggregate(energy_kwh ~ CorrectedTimeOfDay, data=df_analysis, 
                               FUN=function(x) c(mean=mean(x), median=median(x), sd=sd(x)))
print(time_of_day_stats)

# Aggregate statistics by charger type
charger_type_stats <- aggregate(energy_kwh ~ Charger.Type, data=df_analysis, 
                                FUN=function(x) c(mean=mean(x), median=median(x), sd=sd(x)))
print(charger_type_stats)

# Normalizations and transformations

# Identify numeric columns
numeric_cols <- names(df_analysis)[sapply(df_analysis, is.numeric)]

# 1. For Linear Regression and Lasso Regression
## Create a copy
df_linear_lasso <- df_analysis

## Z-score normalization
df_linear_lasso[, numeric_cols] <- scale(df_analysis[, numeric_cols])

for (col in numeric_cols) {
  if (skewness(df_analysis[[col]], na.rm = TRUE) > 0.5) {
    # Store transformed variable with new name
    print(col)
    df_linear_lasso[[paste0(col, "_log")]] <- log(df_analysis[[col]] + 1)
  }
}

# 2. For Random Forest and XGBoost
## Create a copy (usually used without transformations)
df_trees <- df_analysis

## Optional: Transform only highly skewed distributions
for (col in numeric_cols) {
  if (skewness(df_analysis[[col]], na.rm = TRUE) > 2) {
    print(col)
    df_trees[[paste0(col, "_log")]] <- log(df_analysis[[col]] + 1)
  }
}


# 3. Robust Scaling (for data with outliers)
## Create a copy
df_robust <- df_analysis

## Robust Scaling using median and IQR
df_robust[, numeric_cols] <- lapply(df_analysis[, numeric_cols], function(x) {
  (x - median(x, na.rm = TRUE)) / (IQR(x, na.rm = TRUE))
})

## Create a copy
df_boxcox <- df_analysis

## Apply Box-Cox for each positive numeric variable
for (col in numeric_cols) {
  # Check if all values are positive
  if (min(df_analysis[[col]], na.rm = TRUE) > 0) {
    # Calculate optimal lambda
    bc <- powerTransform(df_analysis[[col]])
    lambda <- bc$lambda
    print(col)
    # Apply transformation with optimal lambda
    # Store in new variable to preserve the original
    df_boxcox[[paste0(col, "_bc")]] <- if (abs(lambda) < 0.01) {
      log(df_analysis[[col]])
    } else {
      (df_analysis[[col]]^lambda - 1) / lambda
    }
  }
}

# Check that copies were created correctly
dim(df_analysis)
dim(df_linear_lasso)
dim(df_trees)
dim(df_robust)
dim(df_boxcox)

# Function to create a consistent data split for all datasets
create_data_splits <- function(data, target_var = "energy_kwh") {
  
  # Total data size
  n <- nrow(data)
  
  # Calculate split indices
  train_size <- round(n * 0.7)
  val_size <- round(n * 0.1)
  test1_size <- round(n * 0.1)
  
  # Create indices
  indices <- sample(1:n, n)
  
  # Split the data
  train_indices <- indices[1:train_size]
  val_indices <- indices[(train_size+1):(train_size+val_size)]
  test1_indices <- indices[(train_size+val_size+1):(train_size+val_size+test1_size)]
  test2_indices <- indices[(train_size+val_size+test1_size+1):n]
  
  # Create the splits
  train_data <- data[train_indices, ]
  val_data <- data[val_indices, ]
  test1_data <- data[test1_indices, ]
  test2_data <- data[test2_indices, ]
  
  # Separate features and target
  X_train <- train_data[, !names(train_data) %in% target_var]
  y_train <- train_data[[target_var]]
  
  X_val <- val_data[, !names(val_data) %in% target_var]
  y_val <- val_data[[target_var]]
  
  X_test1 <- test1_data[, !names(test1_data) %in% target_var]
  y_test1 <- test1_data[[target_var]]
  
  X_test2 <- test2_data[, !names(test2_data) %in% target_var]
  y_test2 <- test2_data[[target_var]]
  
  # Return all splits
  return(list(
    X_train = X_train, y_train = y_train,
    X_val = X_val, y_val = y_val,
    X_test1 = X_test1, y_test1 = y_test1,
    X_test2 = X_test2, y_test2 = y_test2,
    train_indices = train_indices,
    val_indices = val_indices,
    test1_indices = test1_indices,
    test2_indices = test2_indices
  ))
}

#------ Linear Regression Model ------#
# Create splits for linear regression dataset
linear_splits <- create_data_splits(df_linear_lasso)

# Train linear regression model
linear_model <- lm(y_train ~ ., data = cbind(linear_splits$X_train, y_train = linear_splits$y_train))

# Make predictions on test set 1
linear_predictions <- predict(linear_model, newdata = linear_splits$X_test1)

# Calculate performance metrics
linear_rmse <- sqrt(mean((linear_predictions - linear_splits$y_test1)^2))
linear_mae <- mean(abs(linear_predictions - linear_splits$y_test1))
linear_r2 <- 1 - sum((linear_predictions - linear_splits$y_test1)^2) / sum((linear_splits$y_test1 - mean(linear_splits$y_test1))^2)
cat("Linear Regression - RMSE:", linear_rmse, "MAE:", linear_mae, "R²:", linear_r2, "\n")

#------ Lasso Regression Model ------#

# Prepare data in matrix format for glmnet
X_train_matrix <- model.matrix(~ ., data = linear_splits$X_train)[, -1]
y_train_vector <- linear_splits$y_train
X_val_matrix <- model.matrix(~ ., data = linear_splits$X_val)[, -1]
y_val_vector <- linear_splits$y_val
X_test1_matrix <- model.matrix(~ ., data = linear_splits$X_test1)[, -1]

# Grid of lambda values
lambda_grid <- 10^seq(-3, 1, length.out=100)

# Cross-validation for optimal lambda
cv_lasso <- cv.glmnet(X_train_matrix, y_train_vector, 
                      alpha = 1, 
                      lambda = lambda_grid,
                      nfolds = 10)

# Plot the CV results
plot(cv_lasso)

# Get the optimal lambda
optimal_lambda <- cv_lasso$lambda.min
cat("Optimal lambda:", optimal_lambda, "\n")

# Try different alpha values (elastic net between ridge and lasso)
alpha_values <- seq(0, 1, by=0.1)
cv_results <- list()

for (alpha in alpha_values) {
  cv_model <- cv.glmnet(X_train_matrix, y_train_vector, 
                        alpha = alpha, 
                        lambda = lambda_grid,
                        nfolds = 10)
  cv_results[[as.character(alpha)]] <- min(cv_model$cvm)
}

# Find best alpha
best_alpha <- alpha_values[which.min(unlist(cv_results))]
cat("Best alpha:", best_alpha, "\n")

# Retrain with optimal parameters
lasso_model <- glmnet(X_train_matrix, y_train_vector, 
                            alpha = best_alpha, 
                            lambda = cv.glmnet(X_train_matrix, y_train_vector, alpha = best_alpha, nfolds = 10)$lambda.min)

# Make predictions on test set 1
lasso_predictions <- predict(lasso_model, s = optimal_lambda, newx = X_test1_matrix)

# Calculate performance metrics
lasso_rmse <- sqrt(mean((lasso_predictions - linear_splits$y_test1)^2))
lasso_mae <- mean(abs(lasso_predictions - linear_splits$y_test1))
lasso_r2 <- 1 - sum((lasso_predictions - linear_splits$y_test1)^2) / sum((linear_splits$y_test1 - mean(linear_splits$y_test1))^2)
cat("Lasso Regression - RMSE:", lasso_rmse, "MAE:", lasso_mae, "R²:", lasso_r2, "\n")

#------ Random Forest Model ------#
# Create splits for tree-based models
trees_splits <- create_data_splits(df_trees)

# Use caret for tuning
control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = FALSE
)

# Parameter grid
rf_grid <- expand.grid(
  mtry = c(2, sqrt(ncol(trees_splits$X_train)), ncol(trees_splits$X_train)/3)
)

# Train model with cross-validation
rf_model <- train(
  x = trees_splits$X_train,
  y = trees_splits$y_train,
  method = "rf",
  trControl = control,
  tuneGrid = rf_grid,
  importance = TRUE,
  ntree = 500
)

# Make predictions on test set 1
rf_predictions <- predict(rf_model, newdata = trees_splits$X_test1)

# Calculate performance metrics
rf_rmse <- sqrt(mean((rf_predictions - trees_splits$y_test1)^2))
rf_mae <- mean(abs(rf_predictions - trees_splits$y_test1))
rf_r2 <- 1 - sum((rf_predictions - trees_splits$y_test1)^2) / sum((trees_splits$y_test1 - mean(trees_splits$y_test1))^2)
cat("Random Forest - RMSE:", rf_rmse, "MAE:", rf_mae, "R²:", rf_r2, "\n")

#------ XGBoost Model ------#
# Prepare data in matrix format for xgboost
X_train_matrix <- model.matrix(~ ., data = trees_splits$X_train)[, -1]
X_val_matrix <- model.matrix(~ ., data = trees_splits$X_val)[, -1]
X_test1_matrix <- model.matrix(~ ., data = trees_splits$X_test1)[, -1]

# Then convert to DMatrix
xgb_train <- xgb.DMatrix(data = X_train_matrix, label = trees_splits$y_train)
xgb_val <- xgb.DMatrix(data = X_val_matrix, label = trees_splits$y_val)
xgb_test1 <- xgb.DMatrix(data = X_test1_matrix)

# Define watchlist for monitoring
watchlist <- list(train = xgb_train, val = xgb_val)

# Parameter grid for grid search
xgb_params <- expand.grid(
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 5, 7),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.7, 0.8, 0.9),
  colsample_bytree = c(0.7, 0.8, 0.9),
  gamma = c(0, 0.1, 0.2)
)

# Function to evaluate XGBoost with specific parameters
evaluate_params <- function(eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma) {
  params <- list(
    objective = "reg:squarederror",
    eta = eta,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    gamma = gamma
  )
  
  cv_results <- xgb.cv(
    params = params,
    data = xgb_train,
    nrounds = 1000,
    nfold = 5,
    early_stopping_rounds = 50,
    metrics = "rmse",
    verbose = 0
  )
  
  return(min(cv_results$evaluation_log$test_rmse_mean))
}

# Run a simplified grid search (for illustration - a full grid would be extensive)
sample_size <- 15  # Taking a sample of parameter combinations
sample_indices <- sample(nrow(xgb_params), sample_size)
sampled_params <- xgb_params[sample_indices, ]

# Evaluate sampled parameters
results <- mapply(
  evaluate_params,
  sampled_params$eta,
  sampled_params$max_depth,
  sampled_params$min_child_weight,
  sampled_params$subsample,
  sampled_params$colsample_bytree,
  sampled_params$gamma
)

# Find best parameters
best_idx <- which.min(results)
best_params <- sampled_params[best_idx, ]
print(best_params)

# Train final model with best parameters
xgb_model <- xgb.train(
  params = list(
    objective = "reg:squarederror",
    eta = best_params$eta,
    max_depth = best_params$max_depth,
    min_child_weight = best_params$min_child_weight,
    subsample = best_params$subsample,
    colsample_bytree = best_params$colsample_bytree,
    gamma = best_params$gamma
  ),
  data = xgb_train,
  nrounds = 1000,
  watchlist = watchlist,
  early_stopping_rounds = 50,
  verbose = 0
)

# Make predictions on test set 1
xgb_predictions <- predict(xgb_model, xgb_test1)

# Calculate performance metrics
xgb_rmse <- sqrt(mean((xgb_predictions - trees_splits$y_test1)^2))
xgb_mae <- mean(abs(xgb_predictions - trees_splits$y_test1))
xgb_r2 <- 1 - sum((xgb_predictions - trees_splits$y_test1)^2) / sum((trees_splits$y_test1 - mean(trees_splits$y_test1))^2)
cat("XGBoost - RMSE:", xgb_rmse, "MAE:", xgb_mae, "R²:", xgb_r2, "\n")

#------ Compare Models ------#
# Create a data frame with model performance metrics
model_comparison <- data.frame(
  Model = c("Linear Regression", "Lasso Regression", "Random Forest", "XGBoost"),
  RMSE = c(linear_rmse, lasso_rmse, rf_rmse, xgb_rmse),
  MAE = c(linear_mae, lasso_mae, rf_mae, xgb_mae),
  R2 = c(linear_r2, lasso_r2, rf_r2, xgb_r2)
)

# Sort by RMSE (lower is better)
model_comparison <- model_comparison[order(model_comparison$RMSE), ]

# Print performance comparison
print(model_comparison)


## 1. Helpers ----
rmse   <- function(a, b) sqrt(mean((a - b)^2))

# generic fit & predict wrappers, re-using tuned hyper-parameters
fit_linear_lc <- function(X, y) lm(y ~ ., data = cbind(X, y))
fit_lasso_lc  <- function(X, y) {
  glmnet::glmnet(model.matrix(~ ., X)[, -1], y,
                 alpha   = best_alpha,
                 lambda  = optimal_lambda)
}
fit_rf_lc <- function(X, y) {
  randomForest::randomForest(x = X, y = y,
                             mtry  = rf_model$bestTune$mtry,
                             ntree = 500)
}
fit_xgb_lc <- function(X, y) {
  dtrain <- xgboost::xgb.DMatrix(model.matrix(~ ., X)[, -1], label = y)
  xgboost::xgb.train(
    params  = list(
      objective        = "reg:squarederror",
      eta              = best_params$eta,
      max_depth        = best_params$max_depth,
      min_child_weight = best_params$min_child_weight,
      subsample        = best_params$subsample,
      colsample_bytree = best_params$colsample_bytree,
      gamma            = best_params$gamma
    ),
    data    = dtrain,
    nrounds = 500,
    verbose = 0)
}
predict_lc <- function(mod, X) {
  if (inherits(mod, "glmnet"))
    as.numeric(predict(mod, model.matrix(~ ., X)[, -1], s = mod$lambda))
  else if (inherits(mod, "xgb.Booster"))
    predict(mod, xgboost::xgb.DMatrix(model.matrix(~ ., X)[, -1]))
  else
    predict(mod, newdata = X)
}

## 2. Learning-curve generator ----
learning_curve <- function(Xtr, ytr, Xval, yval, fit_fun,
                           sizes = seq(0.1, 1, 0.1)) {
  purrr::map_dfr(sizes, function(frac) {
    idx <- sample(seq_len(nrow(Xtr)), round(frac * nrow(Xtr)))
    mod <- fit_fun(Xtr[idx, ], ytr[idx])
    tibble::tibble(
      frac      = frac,
      set       = c("Train", "Validation"),
      rmse      = c(rmse(ytr[idx], predict_lc(mod, Xtr[idx, ])),
                    rmse(yval,     predict_lc(mod, Xval)))
    )
  })
}

## 3. Stability generator ----
stability_curve <- function(X, y, fit_fun, reps = 30, test_p = 0.2) {
  purrr::map_dfr(seq_len(reps), function(r) {
    tst <- sample(seq_len(nrow(X)), round(test_p * nrow(X)))
    mod <- fit_fun(X[-tst, ], y[-tst])
    tibble::tibble(
      iter = r,                       #  <<< changed here
      rmse = rmse(y[tst], predict_lc(mod, X[tst, ]))
    )
  })
}

## 4. Build curves for each model ----
lc_linear <- learning_curve(linear_splits$X_train, linear_splits$y_train,
                            linear_splits$X_val,   linear_splits$y_val,
                            fit_linear_lc)

lc_lasso  <- learning_curve(linear_splits$X_train, linear_splits$y_train,
                            linear_splits$X_val,   linear_splits$y_val,
                            fit_lasso_lc)

lc_rf     <- learning_curve(trees_splits$X_train, trees_splits$y_train,
                            trees_splits$X_val,   trees_splits$y_val,
                            fit_rf_lc)

lc_xgb    <- learning_curve(trees_splits$X_train, trees_splits$y_train,
                            trees_splits$X_val,   trees_splits$y_val,
                            fit_xgb_lc)

stab_linear <- stability_curve(
  rbind(linear_splits$X_train, linear_splits$X_val),
  c(linear_splits$y_train,     linear_splits$y_val),
  fit_linear_lc)

stab_lasso  <- stability_curve(
  rbind(linear_splits$X_train, linear_splits$X_val),
  c(linear_splits$y_train,     linear_splits$y_val),
  fit_lasso_lc)

stab_rf     <- stability_curve(
  rbind(trees_splits$X_train, trees_splits$X_val),
  c(trees_splits$y_train,     trees_splits$y_val),
  fit_rf_lc)

stab_xgb    <- stability_curve(
  rbind(trees_splits$X_train, trees_splits$X_val),
  c(trees_splits$y_train,     trees_splits$y_val),
  fit_xgb_lc)

lc_linear_grp <- bind_rows(
  lc_linear %>% mutate(model = "Linear"),
  lc_lasso  %>% mutate(model = "Lasso")
)

lc_tree_grp <- bind_rows(
  lc_rf  %>% mutate(model = "Random Forest"),
  lc_xgb %>% mutate(model = "XGBoost")
)

plot_lc_group <- function(dat, title) {
  ggplot(dat,
         aes(x = frac, y = rmse,
             colour   = model,
             linetype = set)) +
    geom_line(size = 1) +
    geom_point(size = 1.6) +
    scale_linetype_manual(values = c(Train = "solid",
                                     Validation = "dashed")) +
    labs(title    = title,
         x        = "Training-set fraction",
         y        = "RMSE",
         colour   = "Model",
         linetype = "") +
    theme_minimal()
}

plot_lc_group(lc_linear_grp, "Learning Curves – Linear Models")
plot_lc_group(lc_tree_grp,   "Learning Curves – Tree-based Models")

# helper to make one stability box-plot -----------------------------
make_stab_plot <- function(dat, title) {
  ggplot(dat, aes(x = "", y = rmse)) +
    geom_boxplot(outlier.alpha = 0.4) +
    geom_jitter(width = 0.1, alpha = 0.6) +
    labs(title = title, y = "RMSE across random splits", x = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
}

p_lin  <- make_stab_plot(stab_linear, "Linear Regression")
p_las  <- make_stab_plot(stab_lasso,  "Lasso")
p_rf   <- make_stab_plot(stab_rf,     "Random Forest")
p_xgb  <- make_stab_plot(stab_xgb,    "XGBoost")

# arrange the four plots --------------------------------------------
if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  (p_lin | p_las) /
    (p_rf  | p_xgb) +
    plot_annotation(title = "Model-Stability (RMSE) – 30 random splits each")
} else {                       # fallback to gridExtra if patchwork not present
  gridExtra::grid.arrange(
    p_lin, p_las, p_rf, p_xgb,
    ncol = 2,
    top = "Model-Stability (RMSE) – 30 random splits each"
  )
}


# Identify the best model based on RMSE
best_model_name <- model_comparison$Model[1]
cat("The best performing model based on RMSE is:", best_model_name, "\n")

# To understand why your tree models are underperforming
# For linear model
par(mfrow=c(2,2))
plot(linear_model)


get_vi <- function(model, model_name) {
  
  # 1.  If the model is a caret::train object, fall back to caret::varImp ----
  if ("train" %in% class(model)) {
    vi <- varImp(model, scale = FALSE)$importance      # always a data-frame
    
    if (ncol(vi) == 1) {
      vi <- tibble(Feature = rownames(vi), Raw = vi[, 1])
    } else {                      # multiclass case – average over classes
      vi <- tibble(Feature = rownames(vi), Raw = rowMeans(vi, na.rm = TRUE))
    }
    
    # 2.  Native randomForest or ranger objects --------------------------------
  } else if (inherits(model, "randomForest") ||
             inherits(model, "ranger")) {
    vi <- randomForest::importance(model) %>%          # matrix
      as.data.frame() %>%
      rownames_to_column("Feature") %>%
      rename(Raw = 1)                              # first column is the score
    
    # 3.  XGBoost booster -------------------------------------------------------
  } else if (inherits(model, "xgb.Booster")) {
    vi <- xgboost::xgb.importance(model = model) %>%
      select(Feature, Raw = Gain)
    
    # 4.  Lasso / Elastic-net (glmnet) -----------------------------------------
  } else if (inherits(model, "glmnet")) {
    cf <- as.matrix(coef(model))
    vi <- tibble(Feature = rownames(cf),
                 Raw = abs(as.numeric(cf))) %>%
      filter(Raw > 0)                               # drop zero-coeff vars
    
    # 5.  Linear model – t-statistic magnitude ---------------------------------
  } else if (inherits(model, "lm")) {
    sm <- summary(model)
    vi <- tibble(Feature = names(sm$coefficients)[-1],
                 Raw = abs(sm$coefficients[-1, "t value"]))
  } else {
    stop("No var-importance method for this model class.")
  }
  
  ## rescale 0-1 and add model label
  vi %>%
    mutate(Model = model_name,
           Norm = (Raw - min(Raw)) / (max(Raw) - min(Raw)))
}


vi_rf    <- get_vi(rf_model,    "RF")
vi_xgb   <- get_vi(xgb_model,   "XGB")
vi_lasso <- get_vi(lasso_model, "Lasso")
vi_lm    <- get_vi(linear_model,"Linear")   # optional

imp_all  <- bind_rows(vi_rf, vi_xgb, vi_lasso, vi_lm)   # single tidy frame

print(imp_all, n=Inf)

plot_data <- imp_all %>%
  inner_join(top_feats, by = "Feature")          # keeps MeanImp column

ggplot(plot_data,
       aes(x = Model,
           y = reorder(Feature, MeanImp),        # now MeanImp is found
           fill = Norm)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Cross-model feature importance (normalised)",
       x = NULL, y = NULL, fill = "0-1\nimportance") +
  theme_minimal(base_size = 13)


# Calculate nonconformity scores
nonconformity_scores <- abs(linear_predictions - linear_splits$y_test1)

# Organize scores into empirical distribution (sort them)
sorted_scores <- sort(nonconformity_scores)

# Select quantile for 90% confidence level
alpha <- 0.10
n <- length(sorted_scores)
k <- ceiling((n + 1) * (1 - alpha))
quantile_value <- sorted_scores[k]
cat("Quantile value for 90% confidence:", quantile_value, "\n")

# Generate predictions for test set 2
X_test2_matrix <- model.matrix(~ ., data = linear_splits$X_test2)[, -1]
linear_test2_predictions <- predict(linear_model, newdata = X_test2_matrix)

# Create prediction intervals
lower_bounds <- linear_test2_predictions - quantile_value
upper_bounds <- linear_test2_predictions + quantile_value

# Calculate coverage (percentage of true values within intervals)
actual_test2 <- linear_splits$y_test2
in_interval <- (actual_test2 >= lower_bounds) & (actual_test2 <= upper_bounds)
coverage <- mean(in_interval)
cat("Empirical coverage probability:", coverage, "\n")

# Calculate average interval width
avg_width <- mean(upper_bounds - lower_bounds)
cat("Average interval width:", avg_width, "kWh\n")

#------ Winkler Score for Prediction Intervals ------#
# Compute Winkler score for alpha = 0.10
alpha <- 0.10
# Ensure prediction_data has Actual, Lower, Upper columns
prediction_data$IntervalWidth <- prediction_data$Upper - prediction_data$Lower
prediction_data$WinklerScore <- with(prediction_data, ifelse(
  Actual < Lower,
  IntervalWidth + (2/alpha) * (Lower - Actual),
  ifelse(
    Actual > Upper,
    IntervalWidth + (2/alpha) * (Actual - Upper),
    IntervalWidth
  )
))
# Average Winkler Score
avg_winkler <- mean(prediction_data$WinklerScore)
cat("Average Winkler Score (alpha=", alpha, "):", avg_winkler, "\n")

# Winkler Score by user type
winkler_by_user <- aggregate(WinklerScore ~ UserType, data = prediction_data, FUN = mean)
print("Winkler Score by user type:")
print(winkler_by_user)

# Winkler Score by charger type
winkler_by_charger <- aggregate(WinklerScore ~ ChargerType, data = prediction_data, FUN = mean)
print("Winkler Score by charger type:")
print(winkler_by_charger)

# Winkler Score by time of day
winkler_by_time <- aggregate(WinklerScore ~ TimeOfDay, data = prediction_data, FUN = mean)
print("Winkler Score by time of day:")
print(winkler_by_time)


# Create dataframe for plotting - make sure all variables are explicitly included
prediction_data <- data.frame(
  Index = 1:length(linear_test2_predictions),
  Actual = as.vector(actual_test2),  # Force as vector to avoid dimension issues
  Predicted = as.vector(linear_test2_predictions),
  Lower = as.vector(lower_bounds),
  Upper = as.vector(upper_bounds)
)

prediction_data$InInterval <- (prediction_data$Actual >= prediction_data$Lower) & 
  (prediction_data$Actual <= prediction_data$Upper)

# Verify your dataframe has all required columns
head(prediction_data)

# Then plot
ggplot(prediction_data, aes(x = Index)) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "lightblue", alpha = 0.5) +
  geom_line(aes(y = Predicted), color = "blue") +
  geom_point(aes(y = Actual), color = "red", size = 2) +
  labs(title = "Linear Prediction Intervals (90% Confidence)",
       x = "Observation", y = "Energy Consumed (kWh)") +
  theme_minimal()

# Add relevant features to prediction data for analysis
prediction_data$UserType <- linear_splits$X_test2$User.Type
prediction_data$ChargerType <- linear_splits$X_test2$Charger.Type
prediction_data$TimeOfDay <- linear_splits$X_test2$CorrectedTimeOfDay

# Calculate coverage by user type
coverage_by_user <- aggregate(InInterval ~ UserType, data = prediction_data, FUN = mean)
print("Coverage by user type:")
print(coverage_by_user)

# Calculate coverage by charger type
coverage_by_charger <- aggregate(InInterval ~ ChargerType, data = prediction_data, FUN = mean)
print("Coverage by charger type:")
print(coverage_by_charger)

# Calculate coverage by time of day
coverage_by_time <- aggregate(InInterval ~ TimeOfDay, data = prediction_data, FUN = mean)
print("Coverage by time of day:")
print(coverage_by_time)

# Calculate average interval width by user type
width_by_user <- aggregate(Upper - Lower ~ UserType, data = prediction_data, FUN = mean)
names(width_by_user)[2] <- "AvgWidth"
print("Average interval width by user type:")
print(width_by_user)

# Calculate practical metrics for charging planning
prediction_data$BufferNeeded <- prediction_data$Actual > prediction_data$Upper
prediction_data$OverAllocation <- prediction_data$Lower > prediction_data$Actual
prediction_data$BufferAmount <- pmax(0, prediction_data$Actual - prediction_data$Upper)
prediction_data$OverAmount <- pmax(0, prediction_data$Lower - prediction_data$Actual)

# Summary statistics
buffer_rate <- mean(prediction_data$BufferNeeded)
over_rate <- mean(prediction_data$OverAllocation)
avg_buffer <- mean(prediction_data$BufferAmount)
avg_over <- mean(prediction_data$OverAmount)

#------ False-Positive / False-Negative Analysis ----------------#
# Classify each observation -------------------------------------
prediction_data$PredClass <- with(prediction_data, ifelse(
  BufferNeeded,                        # Actual > Upper
  "FN",                                # under-supply
  ifelse(OverAllocation, "FP", "Covered")  # over-supply vs OK
))

# 2.1  Overall confusion counts ---------------------------------
overall_conf <- table(prediction_data$PredClass)
print("Overall FP / FN / Covered counts:")
print(overall_conf)

# 2.2  Rates -----------------------------------------------------
N <- nrow(prediction_data)
fp_rate <- overall_conf["FP"] / N
fn_rate <- overall_conf["FN"] / N
coverage_rate <- overall_conf["Covered"] / N
cat(sprintf("False-positive rate  : %.2f%%\n", 100*fp_rate))
cat(sprintf("False-negative rate  : %.2f%%\n", 100*fn_rate))
cat(sprintf("Correct-coverage rate: %.2f%%\n", 100*coverage_rate))

# 2.3  By-segment breakdown (example: UserType) -----------------
conf_by_user <- aggregate(
  list(FP = prediction_data$PredClass == "FP",
       FN = prediction_data$PredClass == "FN",
       Covered = prediction_data$PredClass == "Covered"),
  by = list(UserType = prediction_data$UserType),
  FUN = sum
)
conf_by_user$Total <- rowSums(conf_by_user[, c("FP", "FN", "Covered")])
conf_by_user$FP_Rate <- with(conf_by_user, FP / Total)
conf_by_user$FN_Rate <- with(conf_by_user, FN / Total)
conf_by_user$Coverage_Rate <- with(conf_by_user, Covered / Total)

print("Confusion-matrix counts & rates by user type:")
print(conf_by_user[, c("UserType", "FP", "FN", "Covered",
                       "FP_Rate", "FN_Rate", "Coverage_Rate")])

cat("Probability of needing additional energy buffer:", buffer_rate, "\n")
cat("Average additional energy needed when buffer required:", avg_buffer, "kWh\n")
cat("Probability of over-allocating energy:", over_rate, "\n")
cat("Average energy over-allocated when this occurs:", avg_over, "kWh\n")

# Create visualization for decision support
ggplot(prediction_data, aes(x = Predicted, y = Actual)) +
  geom_point(aes(color = BufferNeeded), size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  geom_abline(intercept = quantile_value, slope = 1, linetype = "dotted", color = "blue") +
  geom_abline(intercept = -quantile_value, slope = 1, linetype = "dotted", color = "blue") +
  scale_color_manual(values = c("darkgreen", "red"), 
                     labels = c("Sufficient Allocation", "Buffer Needed")) +
  labs(title = "Energy Allocation Planning with Prediction Intervals",
       x = "Predicted Energy (kWh)", y = "Actual Energy Required (kWh)",
       color = "Energy Planning") +
  theme_minimal()



#------ Performance Metrics by User Type ------#
# Prepare test2 predictions for all models
# Linear Regression predictions on linear_splits$X_test2
linear_test2_preds <- predict(linear_model, newdata = linear_splits$X_test2)
lasso_test2_preds <- as.vector(lasso_test2_predictions)

# Random Forest predictions on trees_splits$X_test2
rf_test2_preds <- predict(rf_model, newdata = trees_splits$X_test2)

# XGBoost predictions on trees_splits$X_test2
X_test2_matrix_trees <- model.matrix(~ ., data = trees_splits$X_test2)[, -1]
xgb_test2_dmatrix <- xgb.DMatrix(data = X_test2_matrix_trees)
xgb_test2_preds <- predict(xgb_model, xgb_test2_dmatrix)

# Actual values for test2
actual_linear_test2 <- linear_splits$y_test2
actual_trees_test2 <- trees_splits$y_test2

# Combine into single data frames for linear-based and tree-based models
eval_data_linear <- data.frame(
  UserType = linear_splits$X_test2$User.Type,
  Actual = actual_linear_test2,
  Linear = linear_test2_preds,
  Lasso = lasso_test2_preds,
  RF = NA,
  XGB = NA
)
eval_data_trees <- data.frame(
  UserType = trees_splits$X_test2$User.Type,
  Actual = actual_trees_test2,
  Linear = NA,
  Lasso = NA,
  RF = rf_test2_preds,
  XGB = xgb_test2_preds
)
eval_data <- rbind(eval_data_linear, eval_data_trees)

# Function to compute metrics given actual and predicted
compute_metrics <- function(actual, pred) {
  idx <- !is.na(pred)
  act <- actual[idx]
  prd <- pred[idx]
  rmse <- sqrt(mean((prd - act)^2))
  mae <- mean(abs(prd - act))
  r2 <- 1 - sum((prd - act)^2) / sum((act - mean(act))^2)
  return(c(RMSE = rmse, MAE = mae, R2 = r2))
}

# List of models to evaluate
models <- c("Linear", "Lasso", "RF", "XGB")

# Compute and display metrics by model and User Type
for (m in models) {
  cat("Performance metrics for", m, "by User Type:\n")
  metrics <- t(sapply(split(eval_data, eval_data$UserType), function(df) {
    compute_metrics(df$Actual, df[[m]])
  }))
  print(metrics)
  cat("\n")
}


segment_summary <- function(df, group_vars,
                            actual_col  = "Actual",
                            pred_col,
                            lower_col   = NULL,
                            upper_col   = NULL) {
  
  width_ok <- !is.null(lower_col) && !is.null(upper_col) &&
    lower_col %in% names(df)   && upper_col %in% names(df)
  
  df %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_vars))) %>%
    dplyr::summarise(
      n_obs = dplyr::n(),
      RMSE  = sqrt(mean( (.data[[actual_col]] - .data[[pred_col]])^2 )),
      MAE   = mean(abs(.data[[actual_col]] - .data[[pred_col]])),
      R2    = 1 - sum((.data[[actual_col]] - .data[[pred_col]])^2) /
        sum((.data[[actual_col]] - mean(.data[[actual_col]]))^2),
      Width = if (width_ok)
        mean(.data[[upper_col]] - .data[[lower_col]])
      else NA_real_,
      .groups = "drop"
    )
}

uncert_charger <- segment_summary(
  df         = prediction_data,
  group_vars = "ChargerType",
  actual_col = "Actual",
  pred_col   = "Predicted",
  lower_col  = "Lower",
  upper_col  = "Upper"
)
print(uncert_charger)

uncert_time <- segment_summary(
  df         = prediction_data,
  group_vars = c("UserType", "TimeOfDay"),
  actual_col = "Actual",
  pred_col   = "Predicted",
  lower_col  = "Lower",
  upper_col  = "Upper"
)
print(uncert_time)












