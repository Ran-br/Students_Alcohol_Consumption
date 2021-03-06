import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def preprocess(data, scaler):

    df = data
    categorical_feature_mask = df.dtypes == object
    categorical_features = df.columns[categorical_feature_mask].tolist()
    numeric_feature_mask = df.dtypes != object
    numeric_features = df.columns[numeric_feature_mask][:-2].tolist()

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=10000))
    ])

    X = data.drop(['T', 'Y'], axis=1)
    T = data['T']
    y = data['Y']
    X_treated = X[data['T'] == 1]
    y_treated = y[data['T'] == 1]
    X_control = X[data['T'] == 0]
    y_control = y[data['T'] == 0]


    clf.fit(X, T)
    pred = clf.predict(X)
    print('Accuracy', clf.score(X, T))
    print('Brier', np.mean((pred - T) ** 2))
    fpr, tpr, _ = roc_curve(T, pred)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr)})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.legend()
    plt.show()



    # categorial = ['x_2', 'x_21', 'x_24']
    # data = data.join(pd.get_dummies(data[categorial]))
    # data = data[data.columns.difference(categorial)]
    # #scaler = MinMaxScaler()
    #
    # scaler = StandardScaler()
    # #scaler.fit(data)
    # #X = scaler.transform(data)
    #
    # X = scaler.fit_transform(data.drop(['T', 'Y'], axis=1))
    # T = data['T']
    # y = data['Y']
    # X_treated = X[data['T'] == 1]
    # y_treated = y[data['T'] == 1]
    # X_control = X[data['T'] == 0]
    # y_control = y[data['T'] == 0]
    return X, T, y, X_treated, y_treated, X_control, y_control


def ipw(data):
    X, T, y, X_treated, y_treated, X_control, y_control = preprocess(data, StandardScaler)
    model = RandomForestClassifier(max_depth=14, n_estimators=10000)
    #model = LogisticRegression(max_iter=10000)
    model.fit(X, T)
    # pred = model.predict(X)
    # print('Accuracy', model.score(X, T))
    # print('Brier', np.mean((pred - T) ** 2))
    # fpr, tpr, _ = roc_curve(T, pred)
    # plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr)})')
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    # plt.legend()
    # plt.show()
    control_e = model.predict_proba(X_control)[:, 1]
    e_ratio = control_e / (1 - control_e)
    return np.mean(y_treated) - np.sum(y_control * e_ratio) / np.sum(e_ratio), model.predict_proba(X)[:, -1]


def s_learner(data):
    X, T, y, X_treated, _, _, _ = preprocess(data)
    model = LinearRegression()
    model.fit(np.c_[X, T], y)
    return np.mean(model.predict(np.c_[X_treated, np.ones(X_treated.shape[0])]) -
                   model.predict(np.c_[X_treated, np.zeros(X_treated.shape[0])]))


def t_learner(data):
    _, _, _, X_treated, y_treated, X_control, y_control = preprocess(data)
    model1 = LinearRegression()
    model0 = LinearRegression()
    model1.fit(X_treated, y_treated)
    model0.fit(X_control, y_control)
    return np.mean(model1.predict(X_treated) - model0.predict(X_treated))


def matching(data):
    _, _, _, X_treated, y_treated, X_control, y_control = preprocess(data)
    knn = KNeighborsRegressor(1)
    knn.fit(X_control, y_control)
    return np.mean(y_treated - knn.predict(X_treated))


def estimate_att(data):
    att_ipw, propensity_scores = ipw(data)
    att = [att_ipw]#, s_learner(data), t_learner(data), matching(data)]
    #return att + [np.mean(att)], propensity_scores
    return att, propensity_scores


if __name__ == '__main__':
    data1 = pd.read_csv('data1.csv')
    data2 = pd.read_csv('data2.csv', index_col=0)
    #att1, propensity1 = estimate_att(data1)
    att2, propensity2 = estimate_att(data2)
    att = pd.DataFrame(data=[np.arange(5) + 1, att1, att2], index=['Type', 'data1', 'data2']).transpose()
    propensity = pd.DataFrame(data=[propensity1, propensity2], index=['data1', 'data2'])
    att['Type'] = att['Type'].astype('int')
    att.to_csv('ATT_results.csv', index=False)
    propensity.to_csv('models_propensity.csv', header=False)
