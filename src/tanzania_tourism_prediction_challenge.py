# -*- coding: utf-8 -*-
"""tanzania-tourism-prediction-challenge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JBitcBZEIznW7ZT6FtZ55qdyLbXYeixf

# Intro

This notebook is a simple and an straightforward solution for the Zindi challenge : [Tanzania Tourism Prediction](https://zindi.africa/competitions/tanzania-tourism-prediction/)

# # Setup
# """

# !pip install -q ydata_profiling

# !mkdir -p assets/ml
# !mkdir -p assets/prediction

# !unzip tanzania-tourism-prediction-challenge.zip -d "assets/dataset"

"""# Solution"""

# Imports


# PATHS
import pickle
import os
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn import datasets
from subprocess import call
DIRPATH = os.path.dirname(os.path.relpath(__file__))  # "."
ASSETS_DIR = os.path.join(DIRPATH, "assets",)
PREDICTIONS_DIR = os.path.join(ASSETS_DIR, "prediction")
DATASET_DIR = os.path.join(ASSETS_DIR, "dataset")
SPECIFIC_DATASET_DIR = os.path.join(
    DATASET_DIR, "tanzania-tourism-prediction-data")
ml_fp = os.path.join(ASSETS_DIR, "ml", "ml_components.pkl")
req_fp = os.path.join(ASSETS_DIR, "ml", "requirements.txt")
eda_report_fp = os.path.join(ASSETS_DIR, "ml", "eda-report.html")

# Download dataset
print(
    f"\n[Info] Download and preparing dataset. \n")
# https://drive.google.com/file/d/1MCsnBd1FdPl7xfO31j2FMGI-J0Rc-54T
call(
    f"gdown 1MCsnBd1FdPl7xfO31j2FMGI-J0Rc-54T  -O '{DATASET_DIR}/' ", shell=True)

call(
    f"unzip -o '{os.path.join(DATASET_DIR, 'tanzania-tourism-prediction-data.zip')}' -d '{DATASET_DIR}/' ", shell=True)

# import some data to play with

train = pd.read_csv(os.path.join(SPECIFIC_DATASET_DIR, 'Train.csv'))
test = pd.read_csv(os.path.join(SPECIFIC_DATASET_DIR, 'Test.csv'))
ss = pd.read_csv(os.path.join(SPECIFIC_DATASET_DIR, 'SampleSubmission.csv'))
print(
    f"\n[Info] Dataset loaded : shape={train.shape}\n{train.head().to_markdown()}\n")

plausible_targets = list(set(train.columns) - set(test.columns))
print(f"\n[Info] Plausible targets : {plausible_targets}\n")
d = {i: j for i, j in enumerate(plausible_targets)}
target_col = plausible_targets[0] if len(plausible_targets) == 1 else int(
    input(f"Select the target column, please : {d}"))


print("\n", train.info(), "\n",)
print("\n", train.describe().to_markdown(), "\n",)

nan_threshold = 0.4
ratio_of_nan = (train.isna().sum()/train.shape[0]).sort_values(ascending=False)
features_under_nan_threshold = ratio_of_nan[ratio_of_nan <
                                            nan_threshold].index.tolist()

print(
    f"\n[Info] Features under the nan threshold  of '{nan_threshold}' (in range 0 1) : {features_under_nan_threshold}\n")

# # pandas profiling
# profile = ProfileReport(train, title="Dataset", html={
#                         'style': {'full_width': True}})
# profile.to_file(eda_report_fp)

# Dataset Splitting : Train -> train, eval
# Please specify features to be ignored
to_ignore_cols = [
    "ID", "Id", "id",
    "date",
    #
    target_col
]


num_cols = list(set(train.select_dtypes('number')) - set(to_ignore_cols))
cat_cols = list(set(train.select_dtypes(
    exclude='number')) - set(to_ignore_cols))
print(f"\n[Info] The '{len(num_cols)}' numeric columns are : {num_cols}\nThe '{len(cat_cols)}' categorical columns are : {cat_cols}")

X, y = train[num_cols+cat_cols], train[target_col].values
X_test = test[num_cols+cat_cols]

X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.1, random_state=0,)

print(
    f"\n[Info] Dataset splitted : (X_train , y_train) = {(X_train.shape , y_train.shape)}, (X_eval, y_eval) = {(X_eval.shape , y_eval.shape)}. \n")


# Modeling

# Imputers
num_imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")
cat_imputer = SimpleImputer(
    strategy="most_frequent").set_output(transform="pandas")

# Scaler & Encoder
cat_ = 'auto'
if len(cat_cols) > 0:
    df_imputed_stacked_cat = cat_imputer.fit_transform(
        pd.concat([train, test], axis=0)[cat_cols])

    cat_ = OneHotEncoder(sparse_output=False, drop="first").fit(
        df_imputed_stacked_cat).categories_

cat_n_uniques = {cat_cols[i]: opts_arr.tolist()
                 for (i, opts_arr) in enumerate(cat_)}
print(
    f"\n[Info] All the available unique values in each category are : {cat_n_uniques}\n")

encoder = OneHotEncoder(categories=cat_, sparse_output=False,
                        drop="first").set_output(transform="pandas")
scaler = StandardScaler().set_output(transform="pandas")

X_train_cat, X_train_num = None, None

if len(cat_cols) > 0:
    X_train_cat = encoder.fit_transform(
        cat_imputer.fit_transform(X_train[cat_cols]))

if len(num_cols) > 0:
    X_train_num = scaler.fit_transform(
        num_imputer.fit_transform(X_train[num_cols]))

X_train_ok = pd.concat([X_train_num, X_train_cat], axis=1)

# Model instanciation
# RandomForestClassifier(random_state=10)
# AdaBoostRegressor, RandomForestRegressor
model = RandomForestRegressor(random_state=10,
                              n_estimators=100,
                              )

# Training
print(
    f"\n[Info] Training.\n[Info] X_train : columns( {X_train.columns.tolist()}), shape: {X_train.shape} .\n")

model.fit(X_train_ok, y_train)

# Evaluation
print(
    f"\n[Info] Evaluation.\n")

X_eval_cat = encoder.transform(
    cat_imputer.transform(X_eval[cat_cols])) if len(cat_cols) > 0 else None

X_eval_num = scaler.transform(
    num_imputer.transform(X_eval[num_cols]))if len(num_cols) > 0 else None

X_eval_ok = pd.concat([X_eval_num, X_eval_cat], axis=1)

y_eval_pred = model.predict(X_eval_ok)


def regression_report(y_true, y_pred):
    """
    """
    metrics = {"mse": [mean_squared_error(y_true, y_pred)],
               "rmse": [mean_squared_error(y_true, y_pred, squared=False)],
               "mae": [mean_absolute_error(y_true, y_pred)],
               }
    return f" REGRESSION REPORT \n{pd.DataFrame(metrics).to_string()}"


print(regression_report(y_eval, y_eval_pred,))


# Exportation
print(
    f"\n[Info] Exportation.\n")
to_export = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "num_imputer": num_imputer,
    "cat_imputer": cat_imputer,
    "scaler": scaler,
    "encoder": encoder,
    "model": model,
}


# save components to file
with open(ml_fp, 'wb') as file:
    pickle.dump(to_export, file)

# Requirements
# ! pip freeze > requirements.txt
call(f"pip freeze > {req_fp}", shell=True)

print(f"[Info] Dictionary to use to as base for dataframe filling :\n", {
      col: [] for col in X_train.columns})

"""# Prediction on the test set"""

X_for_pred = test

X_for_pred_ok = pd.concat([scaler.transform(
    num_imputer.transform(X_for_pred[num_cols]))if len(num_cols) > 0 else None,
    encoder.transform(
        cat_imputer.transform(X_for_pred[cat_cols])) if len(cat_cols) > 0 else None],
    axis=1)

y_pred = model.predict(X_for_pred_ok)

ss.head()

col_1, col_2 = ss.columns.tolist()

# col_1 =
# col_2 =

col_1, col_2

X_for_pred[col_1]

sub = pd.DataFrame(
    {col_1: X_for_pred[col_1],
     col_2: y_pred}
)

sub.head()

sub.hist()

sub_filename = "sub_000.csv"
sub.to_csv(os.path.join(PREDICTIONS_DIR, sub_filename), index=False)

"""The public score on the leaderboard is : 5188894.382"""

try:
    call(f"brew install -q tree", shell=True)
    print("\n\nAssets folder structure")
    call(f"tree {ASSETS_DIR}", shell=True)
except:
    print("You must be on Linux. \nOn Mac replace 'apt-get' by 'brew'")
