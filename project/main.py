import numpy as np
import os
import pandas as pd
# Pearson's correlation feature selection for numeric input and numeric output
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
k_best_values = [10, 50, 100, 300, 500, 1000, 2000, 4000, 8000, 10000, 12000]

## Learning speed
alpha_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000, 10000]

## Max number of iterations
# Some models may not converge when max_iterations is under this value, but these models aren't best performing anyway
max_iters = 100000 # 25000 works well too!

def read_data():
    dataframe_cel_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, 'HTA20_RMA.csv'), index_col=0)

    gestational_age_data = pd.read_csv(os.path.join(RESOURCE_FOLDER, 'anoSC1_v11_nokey_with_test_samples.csv'))
    gestational_age_data['original_order'] = gestational_age_data.index.values

    full_dataframe = pd.merge(dataframe_cel_data.T, gestational_age_data, left_on=dataframe_cel_data.T.index.values,
                            right_on=gestational_age_data.SampleID.values, how='right').sort_values(by='original_order')

    grab_hta_data = full_dataframe['Set'] == 'PRB_HTA'
    return full_dataframe[grab_hta_data].drop(['Platform'], axis=1).drop(['key_0'], axis=1)

def strip_features(full_dataframe):
    is_training = full_dataframe['Train'] == 1
    is_testing = full_dataframe['Train'] == 0
    training_data = full_dataframe[is_training]
    testing_data = full_dataframe[is_testing]

    gestational_age = training_data['GA']
    actual_result = testing_data['GA']
    all_features = training_data.iloc[:, :-6] # grab only the features

    return gestational_age, actual_result, testing_data, all_features

def select_best_features(all_features, gestational_age, k_best_value):
    feature_selector = SelectKBest(score_func=f_regression, k=k_best_value)
    scaler = feature_selector.fit(all_features, gestational_age)

    cols = feature_selector.get_support(indices=True)

    if k_best_value == 2000:
        most_important_features = pd.DataFrame(columns=['Gene', 'Score'])
        for ind in cols:
            new_row = {'Gene':all_features.columns[ind], 'Score':feature_selector.scores_[ind]}
            most_important_features = most_important_features.append(new_row, ignore_index=True)
        most_important_features.to_csv('most_important_features.csv', sep=',', index = None)

    return scaler, scaler.transform(all_features)

def define_models():
    models = dict()

    models['LinearReg'] = LinearRegression()
    models['LinearReg_Normalized'] = LinearRegression(normalize=True)
    for a in alpha_values:
        models['LinearRegression_Ridge_Alpha=' + str(a)] = Ridge(alpha=a, max_iter=max_iters)
        models['LinearRegression_Ridge_Normalized_Alpha=' + str(a)] = Ridge(alpha=a, max_iter=max_iters, normalize=True)
        models['LinearRegression_Lasso_Alpha=' + str(a)] = Lasso(alpha=a, max_iter=max_iters)
        models['LinearRegression_Lasso_Normalized_Alpha=' + str(a)] = Lasso(alpha=a, max_iter=max_iters, normalize=True)
    return models

def evaluate_models(models, best_features, gestational_age, actual_result, all_features, k_best_value):
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

def validate_models(models, best_features, scaler, gestational_age, actual_result, testing_data):
    # Testing all the models

    output_dataframe_validation = pd.DataFrame(columns=['model', 'RMSE'])
    testing_data = testing_data.iloc[:, :-6]

    for model in models:
        models[model].fit(best_features, gestational_age)
        print(model)

        predictions = models[model].predict(scaler.transform(testing_data))

        prediction_dataframe = pd.DataFrame(columns=['Predictions'])
        prediction_dataframe.index += 367
        for outcome in predictions:
            prediction_dataframe.loc[-1] = [outcome]
            prediction_dataframe.index += 1

        rmse = np.sqrt(((prediction_dataframe.to_numpy() - actual_result.to_numpy()) ** 2).mean())
        output_dataframe_validation.loc[-1] = [model] + [rmse]
        output_dataframe_validation.index += 1

    output_dataframe_validation = output_dataframe_validation.sort_values(by=['RMSE'])
    print(output_dataframe_validation)

def main():
    full_dataframe = read_data()
    gestational_age, actual_result, testing_data, all_features = strip_features(full_dataframe)

    models = define_models()

    for k_best_value in k_best_values:
        scaler, best_features = select_best_features(all_features, gestational_age, k_best_value)
        print(f"Running evaluation and validation for K value: {k_best_value}")
        # evaluate_models(models, best_features, gestational_age, actual_result, all_features, k_best_value)
        # validate_models(models, best_features, scaler, gestational_age, actual_result, testing_data)
        print("-------------------------------------------------")

if __name__ == "__main__":
    main()
