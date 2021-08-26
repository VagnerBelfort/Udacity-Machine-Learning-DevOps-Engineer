"""
A module read, analyze and model the data to predict churn

Author: Vagner Belfort
Date: August 2021
"""

import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data = pd.read_csv(pth)
    data['Churn'] = data['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return data


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    '''
    columns_plot = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct", "corr_df"]
    for column in columns_plot:
        plt.figure(figsize=(20, 10))
        if column == "Churn":
            data_frame["Churn"].hist()
        elif column == "Customer_Age":
            data_frame["Customer_Age"].hist()
        elif column == "Marital_Status":
            data_frame.Marital_Status.value_counts("normalize").plot(kind="bar")
        elif column == "Total_Trans_Ct":
            sns.displot(data_frame["Total_Trans_Ct"])
        elif column == "corr_df":
            sns.heatmap(data_frame.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig("images/eda/%s.png" % column)


def encoder_helper(data_frame, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    for column_category in category_lst:
        encoded_list = []
        encoded_groups = data_frame.groupby(column_category).mean()['Churn']
        for val in data_frame[column_category]:
            encoded_list.append(encoded_groups.loc[val])
        data_frame[column_category + "_" + "Churn"] = encoded_list
    return data_frame


def perform_feature_engineering(data_frame):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_data = data_frame['Churn']
    x_data = pd.DataFrame()

    columns = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
    x_data[columns] = data_frame[columns]
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data, test_size= 0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    classification_reports_data = {
        "Random_Forest": (
            "Random Forest Train",
            y_test,
            y_test_preds_rf,
            "Random Forest Test",
            y_train,
            y_train_preds_rf),
        "Logistic_Regression": (
            "Logistic Regression Train",
            y_train,
            y_train_preds_lr,
            "Logistic Regression Test",
            y_test,
            y_test_preds_lr)}
    for title, classification_data in classification_reports_data.items():
        plt.rc("figure", figsize=(5, 5))
        plt.text(0.01, 1.25, str(classification_data[0]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    classification_data[1], classification_data[2])), {
                "fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.6, str(classification_data[3]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    classification_data[4], classification_data[5])), {
                "fontsize": 10}, fontproperties="monospace")
        plt.axis("off")
        plt.savefig("images/results/" + title + ".png")


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig("images/" + output_pth + "/Feature_Importance.png")


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, x_test, "results")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    df_churn = import_data("data/bank_data.csv")
    perform_eda(df_churn)

    list_column_category = ["Gender", "Education_Level",
                            "Marital_Status", "Income_Category", "Card_Category"]
    encoded_df_churn = encoder_helper(df_churn, list_column_category)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(encoded_df_churn)

    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
