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
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
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


# Split the data to all the separate components
def split_data(df_data):
    X = df_data.drop(['T', 'Y'], axis=1)
    T, y = df_data['T'], df_data['Y']
    X_treated, X_control = X[df_data['T'] == 1], X[df_data['T'] == 0]
    y_treated, y_control = y[df_data['T'] == 1], y[df_data['T'] == 0]

    return X, X_treated, X_control, y, y_treated, y_control, T


# Wrap all the preprocessing in a preprocessor object
def get_preprocessor(df):
    # Find and split all columns to categorical and numerical
    categorical_feature_mask = df.dtypes == object
    categorical_features = df.columns[categorical_feature_mask].tolist()
    numeric_feature_mask = df.dtypes != object
    numeric_features = df.columns[numeric_feature_mask][:-2].tolist()

    # One-hot encode the categorical values
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Scale the numeric values
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
    ])

    # Join all columns back together
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


# Build a pipeline to wrap the pre-processing
def get_pipeline(df, scaler_type):
    # Find and split all columns to categorical and numerical
    categorical_feature_mask = df.dtypes == object
    categorical_features = df.columns[categorical_feature_mask].tolist()
    numeric_feature_mask = df.dtypes != object
    numeric_features = df.columns[numeric_feature_mask][:-2].tolist()

    # One-hot encode the categorical values
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Scale the numeric values
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
    ])

    # Join all columns back together
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create the outer pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    return pipeline

def calc_propensity(df, pipeline):
    X, X_treated, X_control, y, y_treated, y_control, T = split_data(df)
    pipeline = get_pipeline(df)
    # pipeline2 = get_pipeline(df_data2, StandardScaler)
    pipeline.steps.append(('clf', LogisticRegression(max_iter=1000)))
    pipeline.fit(X, T)
    pred = pipeline.predict(X)
    print('Accuracy', pipeline.score(X, T))
    print('Brier', np.mean((pred - T) ** 2))
    fpr, tpr, _ = roc_curve(T, pred)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr)})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.legend()
    plt.show()
    return 1


# def process_data(data, scaler = MinMaxScaler):
#     categories_col = ['x_2', 'x_21', 'x_24']
#     categories_data = data[categories_col]
#     categories_data = pd.get_dummies(categories_data)
#     data = pd.concat([data, categories_data], axis=1)
#     data = data.drop(categories_col,axis=1)
#     # categories_data = pd.DataFrame(categories_data, columns=model.classes_)
#     scaler.fit_transform(data)
#     print(data)
#
#     return []
#
#     # standardization_model = MinMaxScaler()


# def propensity_score(X, Y):
#     X, T, y, X_treated, y_treated, X_control, y_control = preprocess(data, MinMaxScaler)
#     model = RandomForestClassifier(max_depth=14, n_estimators=200)
#     # model = LogisticRegression(max_iter=10000)
#     model.fit(X, T)
#     pred = model.predict(X)
#     print('Accuracy', model.score(X, T))
#     print('Brier', np.mean((pred - T) ** 2))
#     fpr, tpr, _ = roc_curve(T, pred)
#     plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr)})')
#     plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
#     plt.legend()
#     plt.show()
#     control_e = model.predict_proba(X_control)[:, 1]
#     e_ratio = control_e / (1 - control_e)
#     return np.mean(y_treated) - np.sum(y_control * e_ratio) / np.sum(e_ratio), model.predict_proba(X)[:, -1]


def main():
    df_data1 = pd.read_csv('data1.csv', index_col=0)
    df_data2 = pd.read_csv('data2.csv', index_col=0)

    estimator1 = ATT_Estimator(df_data1)
    estimator2 = ATT_Estimator(df_data2)
    estimator1.calc_propensity()
    estimator2.calc_propensity()
    print("IPW_1: ", estimator1.ipw_att())
    print("IPW_2: ", estimator2.ipw_att())
    # pipeline1 = get_pipeline(df_data1, StandardScaler)
    # pipeline2 = get_pipeline(df_data2, StandardScaler)

    # calc_propensity(df_data1, pipeline1)
    # calc_propensity(df_data2, pipeline2)



    #preprocessor = get_preprocessor(df, MinMaxScaler)




    #x = process_data(data1)
    # data1_prop_score =
    # data2_prop_score =


if __name__ == '__main__':
    main()
