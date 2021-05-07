import pandas as pd
from deeplearning import *
import os

df = pd.read_csv('datasets/iris_with_MV.csv')
imputed = ngram(df)
evaluation(imputed)