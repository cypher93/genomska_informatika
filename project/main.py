# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# pearson's correlation feature selection for numeric input and numeric output
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import PolynomialFeatures

import time

RESOURCE_FOLDER = 'resources'

# List of hyper parameters:
## k - in k-fold cross-validation (set to 10)

## K - in SelectKBest - selecting the most useful features
k_best_value = 8000

## Learning speed
alpha_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000, 10000]

# Step 1 - read data set (y with ID and all features - X with ID)
# Step 2 - grab most useful features using Pearson's correlation coefficient
# Step 3 skipped: remove outliers
# Step 4 - start k-fold cross-validation studying and evaluation. Regression + regularization
# Step 5 - Test it out and print results!

########################
## Let's begin        ##
## Step 1 - read data ##
########################

dataframe_cel_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, 'HTA20_RMA.csv'), index_col=0)

gestational_age_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, 'anoSC1_v11_nokey_with_test_samples.csv'))
gestational_age_data['original_order'] = gestational_age_data.index.values

full_dataframe = pd.merge(dataframe_cel_data.T, gestational_age_data, left_on=dataframe_cel_data.T.index.values,
                          right_on=gestational_age_data.SampleID.values, how='right').sort_values(by='original_order')

grab_hta_data = full_dataframe['Set'] == 'PRB_HTA'
full_dataframe = full_dataframe[grab_hta_data]
full_dataframe = full_dataframe.drop(['Platform'], axis=1).drop(['key_0'], axis=1)

#############################
## Step 2 - strip features ##
#############################

is_training = full_dataframe['Train'] == 1
is_testing = full_dataframe['Train'] == 0
training_data = full_dataframe[is_training]
testing_data = full_dataframe[is_testing]

gestational_age = training_data['GA']
actual_result = testing_data['GA']

all_features = training_data.iloc[:, :-6] # grab only the features

# feature_selector = SelectKBest(score_func=f_regression, k=k_best_value).fit_transform(all_features, gestational_age)

feature_selector = SelectKBest(score_func=f_regression, k=k_best_value)
scaler = feature_selector.fit(all_features, gestational_age)
best_features = scaler.transform(all_features)

# Get columns to keep and create new dataframe with those only
# cols = feature_selector.get_support(indices=True)
# selected_features = all_features.iloc[:, cols]
# print(selected_features)
# print(selected_features.shape)
# print(selected_features.columns)
# selected_features = fs.fit_transform(all_features, gestational_age)

###########################################################################
## Step 4 - define models, k-fold cross-validate and display performance ##
###########################################################################

### Define a bunch of models with different parameters
models = dict()

models['LinearReg'] = LinearRegression()
models['LinearReg_Normalized'] = LinearRegression(normalize=True)
for a in alpha_values:
    models['LinearRegression_Ridge_Alpha=' + str(a)] = Ridge(alpha=a)
    models['LinearRegression_Ridge_Normalized_Alpha=' + str(a)] = Ridge(alpha=a, normalize=True)
    models['LinearRegression_Lasso_Alpha=' + str(a)] = Lasso(alpha=a)
    models['LinearRegression_Lasso_Normalized_Alpha=' + str(a)] = Lasso(alpha=a, normalize=True)

# Create output dataframe
output_dataframe = pd.DataFrame(columns=['model', 'RMSE'])

# Train and validate
for model in models:
    # Split data - K-fold cross-validation
    # random_state gives us the same split every time... In practice, maybe not use it, but good for evaluation
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    cross_validation_results = cross_val_score(models[model], best_features, gestational_age,
                               cv=kfold, scoring='neg_root_mean_squared_error')

    rmse = np.mean(np.abs(cross_validation_results))
    output_dataframe.loc[-1] = [model] + [rmse]
    output_dataframe.index += 1

# Sort by RMSE - best performing on top
output_dataframe = output_dataframe.sort_values(by=['RMSE'])
print(output_dataframe)

#####################################################
## Step 5 - Run best performing model on test data ##
#####################################################

# Testing the model
best_model = models['LinearReg_Normalized'].fit(best_features, gestational_age)

# predict something
testing_data = testing_data.iloc[:, :-6]
predictions = best_model.predict(scaler.transform(testing_data))

prediction_dataframe = pd.DataFrame(columns=['Predictions'])
prediction_dataframe.index += 367
for outcome in predictions:
    prediction_dataframe.loc[-1] = [outcome]
    prediction_dataframe.index += 1

# prediction_dataframe.to_csv('predicted.tsv', sep='\t', index = None)
# actual_result.to_csv('actual.tsv', sep='\t', index = None)

print(np.sqrt(((prediction_dataframe.to_numpy() - actual_result.to_numpy()) ** 2).mean()))
