import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import Lasso

CATEGORICAL_FEATURES = ['Customer Name', 'Customer e-mail', 'Country']
DETERMINERS = ['Gender','Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']

def read_file_csv(filename, delimiter):
    data_frame = pd.read_csv(filename, sep=delimiter, encoding='latin-1')
    return data_frame

def data_split_to_influencers_and_convinced(dataframe):
    df_in = dataframe
    df_co = dataframe[dataframe.columns.difference(DETERMINERS)]
    return df_in,df_co

if __name__ == '__main__':
    # delimiter of prediction csv file
    delimiter = ','

    # file name
    file_name = 'Customer Data.csv'

    # read dataframe
    dataframe = read_file_csv(file_name, delimiter)

    # split data for influencers,convinced
    dataframe_influencers,dataframe_convinced = data_split_to_influencers_and_convinced(dataframe)

    # split data for training and testing data
    training_dataframe_influencers, test__dataframe_influencers, training_dataframe_convinced,test_dataframe_convinced = train_test_split(dataframe_influencers,dataframe_convinced,test_size=0.2)

    training_dataframe_influencers = training_dataframe_influencers[DETERMINERS]
    training_dataframe_convinced = training_dataframe_convinced[training_dataframe_convinced.columns.difference(CATEGORICAL_FEATURES)]

    test__dataframe_influencers = test__dataframe_influencers[DETERMINERS]
    test_dataframe_convinced_to_concat = test_dataframe_convinced[CATEGORICAL_FEATURES]
    test_dataframe_convinced = test_dataframe_convinced[test_dataframe_convinced.columns.difference(CATEGORICAL_FEATURES)]



    # --------------------------------------------------------------------------------
    # model_training_random_forest
    regressor_rf = RandomForestRegressor(n_estimators=100)
    regressor_rf.fit(training_dataframe_influencers,training_dataframe_convinced)

    # model prediction_random_forest
    Y_pred = regressor_rf.predict(test__dataframe_influencers)
    test_dataframe_convinced['prediction of loan amount'] = Y_pred

    #concat
    prediction_dataframe_rf = pd.concat([test_dataframe_convinced_to_concat, test_dataframe_convinced], axis=1).reset_index(drop=True)

    # r2 value
    r2_value = r2_score(test_dataframe_convinced['Loan Amount'],
             test_dataframe_convinced['prediction of loan amount'])
    print('Random forest')
    print(r2_value)
    # ---------------------------------------------------------------------------------
    # model_training_Linear Regression
    regressor_linearReg = linear_model.LinearRegression()
    regressor_linearReg.fit(training_dataframe_influencers, training_dataframe_convinced)

    # model prediction_Linear Regression
    Y_pred = regressor_linearReg.predict(test__dataframe_influencers)
    test_dataframe_convinced['prediction of loan amount'] = Y_pred

    # concat
    prediction_dataframe_lr = pd.concat([test_dataframe_convinced_to_concat, test_dataframe_convinced],
                                     axis=1).reset_index(drop=True)

    # r2 value
    r2_value = r2_score(test_dataframe_convinced['Loan Amount'],
                        test_dataframe_convinced['prediction of loan amount'])
    print('Linear Regression')
    print(r2_value)
    # ---------------------------------------------------------------------------------
    # model_training_Lasso Regression
    regressor_lasso = Lasso(alpha=0.01, max_iter=10e5)
    regressor_lasso.fit(training_dataframe_influencers, training_dataframe_convinced)

    # model prediction_Lasso Regression
    Y_pred = regressor_lasso.predict(test__dataframe_influencers)
    test_dataframe_convinced['prediction of loan amount'] = Y_pred

    # concat
    prediction_dataframe_lasr = pd.concat([test_dataframe_convinced_to_concat, test_dataframe_convinced],
                                     axis=1).reset_index(drop=True)

    # r2 value
    r2_value = r2_score(test_dataframe_convinced['Loan Amount'],
                        test_dataframe_convinced['prediction of loan amount'])
    print('Lasso Regression')
    print(r2_value)
    # ---------------------------------------------------------------------------------