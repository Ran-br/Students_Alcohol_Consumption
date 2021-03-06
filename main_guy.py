import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression, LogisticRegression

def split_and_preprocess(df):
    # Find and split all columns to categorical and numerical
    categorical_feature_mask = df.dtypes == object
    categorical_features = df.columns[categorical_feature_mask].tolist()
    numeric_feature_mask = df.dtypes != object
    numeric_features = df.columns[numeric_feature_mask].tolist()

    df = df[numeric_features].join(pd.get_dummies(df[categorical_features]))

    X = StandardScaler().fit_transform(df.drop(['T', 'Y'], axis=1))
    T, y = df['T'], df['Y']
    X_treated, X_control = X[df['T'] == 1], X[df['T'] == 0]
    y_treated, y_control = y[df['T'] == 1], y[df['T'] == 0]

    return X, X_treated, X_control, y, y_treated, y_control, T

class ATT_Estimator:
    def __init__(self, data):
        df = data
        (self.X,
         self.X_treated,
         self.X_control,
         self.y,
         self.y_treated,
         self.y_control,
         self.T) = split_and_preprocess(data)
        self.prop_model = None

    def calc_propensity(self):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.T)
        pred = model.predict(self.X)
        print('Accuracy', model.score(self.X, self.T))
        print('Brier', np.mean((pred - self.T) ** 2))
        fpr, tpr, _ = roc_curve(self.T, pred)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr)})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='LR Classifier')
        plt.legend()
        plt.show()

        self.prop_model = model

    def propensity_score(self, X_values):
        return self.prop_model.predict_proba(X_values)[:, -1]

    def ipw_att(self):
        # Calc the propensity of the control group (not treated)
        e_x = self.propensity_score(self.X_control)
        weighted_e_x = e_x / (1 - e_x)

        # Use the formula from class to calculate ATT with IPW
        att = (np.sum(self.y_treated)/len(self.y_treated) -
               np.sum(self.y_control * weighted_e_x) / np.sum(weighted_e_x))
        return att

# Main starts here
def main():
    df_data1 = pd.read_csv('data1.csv', index_col=0)
    df_data2 = pd.read_csv('data2.csv', index_col=0)

    estimator1 = ATT_Estimator(df_data1)
    estimator2 = ATT_Estimator(df_data2)
    estimator1.calc_propensity()
    estimator2.calc_propensity()
    print("IPW_1: ", estimator1.ipw_att())
    print("IPW_2: ", estimator2.ipw_att())

if __name__ == '__main__':
    main()
