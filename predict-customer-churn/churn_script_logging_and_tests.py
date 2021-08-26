"""
A module to test churn_library.py

Author: Vagner Belfort
Date: August 2021
"""

import os
import logging
import joblib
from churn_library import import_data, perform_eda, encoder_helper
from churn_library import train_models, perform_feature_engineering
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(data_frame):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data(data_frame)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return data_frame


def test_eda(data_frame):
    '''
    test perform eda function
    '''
    perform_eda(data_frame)
    for image in [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
            "corr_df"]:
        try:
            with open("images/eda/" + image + ".png"):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError:
            logging.error("Testing perform_eda: images not found")


def test_encoder_helper(dataframe):
    '''
    test encoder helper
    '''
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: DataFrame does not contain rows or columns")
        raise err

    try:
        dataframe = encoder_helper(dataframe,
                                   ["Gender",
                                    "Education_Level",
                                    "Marital_Status",
                                    "Income_Category",
                                    "Card_Category"])
        logging.info("Testing test_encoder_helper: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing encoder_helper: Columns do not exist in DataFrame")
        raise err

    return dataframe


def test_perform_feature_engineering(dataframe):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing feature_engineering: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing feature_engineering: The training and test division is incorrect")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(x_train, x_test, y_train, y_test)
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing test_train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: trained models not found")
        raise err

    for image in [
        "Feature_Importance",
        "Logistic_Regression",
            "Random_Forest"]:
        try:
            with open("images/results/" + image + ".png", 'r'):
                logging.info(
                    "Testing testing_models (result of models): SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing test_train_models (result of models): results images not found")
            raise err


if __name__ == "__main__":
    df_churn = test_import("./data/bank_data.csv")
    test_eda(df_churn)

    encoded_df_churn = test_encoder_helper(df_churn)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        encoded_df_churn)

    test_train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
