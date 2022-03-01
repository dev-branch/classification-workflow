import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Cleaner:
    def __init__(self, cols_to_save):
        self.cols_to_save = cols_to_save

    def fit(self, X_train):
        self.mean_age = X_train.age.mean()
        self.mode_embarked = X_train.embarked.mode()[0]

        temp_df = X_train.copy() # so the encoder will not learn about the NaN values
        temp_df.embarked = temp_df.embarked.fillna(self.mode_embarked)
        self.enc = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        self.enc.fit(temp_df[['embarked']])

    def transform(self, X_any):
        df = X_any[self.cols_to_save].copy()
        df = df.reset_index(drop=True)
        df.age = df.age.fillna(self.mean_age)
        df['is_male'] = df.sex == 'male'
        df.embarked = df.embarked.fillna(self.mode_embarked)
        emb_df = pd.DataFrame(self.enc.transform(df[['embarked']]), columns=self.enc.get_feature_names_out())
        df = df.drop(['sex', 'embarked'], axis=1)
        return pd.concat([df, emb_df], axis=1)
