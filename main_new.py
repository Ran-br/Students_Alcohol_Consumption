import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors._dist_metrics import DistanceMetric

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor


def split_and_preprocess(df):
    # Find and split all columns to categorical and numerical
    categorical_feature_mask = df.dtypes == object
    categorical_features = df.columns[categorical_feature_mask].tolist()
    numeric_feature_mask = df.dtypes != object
    numeric_features = df.columns[numeric_feature_mask].tolist()

    df = df[numeric_features].join(pd.get_dummies(df[categorical_features]))

    X = MinMaxScaler().fit_transform(df.drop(['T', 'Y'], axis=1))
    T, y = df['T'], df['Y']
    X_treated, X_control = X[df['T'] == 1], X[df['T'] == 0]
    y_treated, y_control = y[df['T'] == 1], y[df['T'] == 0]

    return X, X_treated, X_control, y, y_treated, y_control, T

class ATT_Estimator:
    def __init__(self, data):
        self.df = data
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
        # print('Accuracy', model.score(self.X, self.T))
        # print('Brier', np.mean((pred - self.T) ** 2))
        # fpr, tpr, _ = roc_curve(self.T, pred)
        # plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr)})')
        # plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        # plt.legend()
        # plt.show()

        self.prop_model = model

    def propensity_score(self, X_values):
        return self.prop_model.predict_proba(X_values)[:, -1]

    def ipw_att(self):
        # Calc the propensity of the control group (not treated)
        e_x = self.propensity_score(self.X_control)
        weighted_e_x = e_x / (1 - e_x)

        # Use the formula from class to calculate ATT with IPW
        att = (np.sum(self.y_treated) / len(self.y_treated) -
               np.sum(self.y_control * weighted_e_x) / np.sum(weighted_e_x))
        return att

    def s_learner_att(self, model=LinearRegression()):
        # print(self.X_treated)
        # print(self.T)
        # model = LinearRegression()
        # x= np.column_stack([self.X, self.T])
        # model = LinearRegression()
        model.fit(np.column_stack([self.X, self.T]), self.y)
        # ATT-> Predict on the treated group only
        y_predict_treated = model.predict(np.column_stack([self.X_treated, np.ones(self.X_treated.shape[0])]))
        y_predict_not_treated = model.predict(np.column_stack([self.X_treated, np.zeros(self.X_treated.shape[0])]))
        return np.mean(y_predict_treated-y_predict_not_treated)

    def s_learner_att_2d(self, model=LinearRegression()):
        T = self.T.to_numpy().reshape([-1, 1])[self.df['T'] == 1]
        print(self.X)
        print(self.X_treated.shape)
        X_extended = np.column_stack([self.X_treated, self.X_treated * T])
        print(X_extended)
        model.fit(np.column_stack([X_extended, T]), self.y_treated)
        X_treated_extended = X_extended
        print(X_treated_extended.shape)
        print(np.column_stack([X_treated_extended, np.ones(X_treated_extended.shape[0])]))
        # print(np.column_stack([X_treated, np.ones(self.X_treated.shape[0])]))
        y_predict_treated = model.predict(np.column_stack([X_treated_extended, np.ones(X_treated_extended.shape[0])]))
        y_predict_not_treated = model.predict(np.column_stack([X_treated_extended, np.zeros(X_treated_extended.shape[0])]))
        return np.mean(y_predict_treated - y_predict_not_treated)

    def matching_att(self):
        model = KNeighborsRegressor(algorithm='brute', metric='mahalanobis', n_neighbors=1)

        # Propensity Score Matching
        # control_propensity = self.propensity_score(self.X_control).reshape([-1, 1])
        # treated_propensity = self.propensity_score(self.X_treated).reshape([-1, 1])
        # model.fit(control_propensity, self.y_control)
        # y_predicted = model.predict(treated_propensity)

        # Train the model all the data
        model.fit(self.X_control, self.y_control)

        # Predict for treated only
        y_predicted = model.predict(self.X_treated)

        return np.mean(self.y_treated - y_predicted)

    def matching(self):
        # from pymatch.Matcher import Matcher
        #
        # m = Matcher(self.X_treated, self.X_control, yvar="Y", exclude=[])
        # print(m)

        dist = DistanceMetric.get_metric('euclidean')
        distances = dist.pairwise(self.X_control, self.X_treated)
        matches = {}
        y_diff = []

        match_dict = {}

        median = np.median(distances)
        for idx, val in enumerate(self.X_control):
            min_dist = distances[idx].min()

            if min_dist <= median:
                nearest_idx = distances[idx].argmin()
                # distances = np.delete(distances, idx, 1)

                match_dict[nearest_idx] = match_dict[nearest_idx]+1 if nearest_idx in match_dict else 1

                matches[idx] = min_dist
                y_diff.append(self.y_treated.iloc[nearest_idx] - self.y_control.iloc[idx])

        return np.mean(y_diff)
        #
        # treated = self.df[self.df['T'] == 1]
        # control = self.df[self.df['T'] == 0]
        # matching_order = np.random.permutation(self.df[self.df['T'] == 1].index)
        # matches = {}
        #
        # for obs in matching_order:
        #     distance = abs(treated[treated["index"] == obs] - control[obs])
        #
        #     if distance.min() <= 0.05:
        #         matches[obs] = [distance.argmin()]




# Main starts here
def main():
    df_data1 = pd.read_csv('data1.csv', index_col=0)
    df_data2 = pd.read_csv('data2.csv', index_col=0)

    estimator1 = ATT_Estimator(df_data1)
    estimator2 = ATT_Estimator(df_data2)
    estimator1.calc_propensity()
    estimator2.calc_propensity()
    # estimator1.s_learner_att_2d()
    print("IPW_1: ", estimator1.ipw_att())
    # print("IPW_2: ", estimator2.ipw_att())
    print("s-learner:  ", estimator1.s_learner_att())
    print("s-learner 2d+1:  ", estimator1.s_learner_att_2d())
    # print("Matching1 :", estimator1.matching_att())
    # print("Matching2 :", estimator2.matching_att())

    print("matching ATT: ", estimator1.matching())


if __name__ == '__main__':
    main()
