import pandas as pd
from sklearn.model_selection import train_test_split


def get_raw_data():
    df = pd.read_csv('data/titanic.csv')
    y = df.Survived
    X = df.drop(['Survived'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def fix_col_names(df):
    df = df.copy()
    df.columns = df.columns.str.replace(r' ', '_').str.lower()
    return df
