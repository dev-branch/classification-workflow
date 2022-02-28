import pandas as pd


class Cleaner:
    def __init__(self, cols_to_save):
        self.cols_to_save = cols_to_save

    def fit(self, X_train):
        self.mean_age = X_train.age.mean()
        self.mode_embarked = X_train.embarked.mode()[0]

    def transform(self, X_any):
        df = X_any[self.cols_to_save].copy()
        df.age = df.age.fillna(self.mean_age)
        df['is_male'] = df.sex == 'male'
        df.embarked = df.embarked.fillna(self.mode_embarked)
        df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

        df = df.drop(['sex'], axis=1)
        return df
