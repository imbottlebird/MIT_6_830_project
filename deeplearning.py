import datawig
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def ngram(df):
    df_drop = df.dropna()
    df_train, df_test = train_test_split(df_drop, test_size=0.15, random_state=RANDOM_SEED)
    #df_train, df_test = datawig.utils.random_split(df)

    #Initialize a SimpleImputer model
    imputer = datawig.SimpleImputer(
        input_columns=['sepal length','sepal width','petal width','class'], # column(s) containing information about the column we want to impute
        output_column='petal length', # the column we'd like to impute values for
        #output_path = 'imputer_model', # stores model data and metrics
        )

    #hyperparameter opimization for numerical data
    imputer.fit_hpo(
        train_df=df_train,
        num_epochs=10,
        learning_rate_candidates=[1e-3, 1e-4],
        final_fc_hidden_units=[[100]]
    )
    #Fit an imputer model on the train data
    imputer.fit(train_df=df_train)

    #Impute missing values and return original dataframe with predictions
    df_mv_list = df[df['petal length'].isnull()]
    imputed_mv = imputer.predict(df_mv_list)

    #bring and map actual values from the original data
    imputed_mv['actual'] = imputed_mv.index.map(df.set_index(df.index)['petal length'])
    return imputed_mv


def evaluation(imputed_mv):
    baseline = np.random.uniform(
    low = min(imputed_mv['petal length']),
    high = max(imputed_mv['petal length']),
    size = len(imputed_mv))

    MSE = [np.square(np.subtract(imputed_mv['actual'],
                                imputed_mv['petal length_imputed'])).mean(),
        np.square(np.subtract(imputed_mv['actual'],
                                baseline)).mean()]
    
    RMSE1 = math.sqrt(MSE[0])
    RMSE2 = math.sqrt(MSE[1])
    print("Root Mean Square Error\n")
    print("Predicted (n-gram): ", RMSE1)
    print("Baseline (random imputation)", RMSE2)

