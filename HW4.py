import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

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

def r_square_graph(y_predicted, y, model_type):
    fig, ax = plt.subplots()
    ax.scatter(y, y_predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'R2: {model_type}')
    plt.show()


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
        self.prop_model = self.calc_propensity()
        self.all_propensity_score = self.propensity_score(self.X)

    def calc_propensity(self):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.T)
        return model

    def propensity_score(self, X_values):
        return self.prop_model.predict_proba(X_values)[:, -1]

    def ipw_att(self):
        # Calc the propensity of the control group
        e_x = self.propensity_score(self.X_control)
        weighted_e_x = e_x / (1 - e_x)

        # Use the formula from class to calculate ATT with IPW
        att = (np.sum(self.y_treated) / len(self.y_treated) -
               np.sum(self.y_control * weighted_e_x) / np.sum(weighted_e_x))
        return att

    def s_learner_att(self, model=GaussianProcessRegressor()):
        model.fit(np.column_stack([self.X, self.T]), self.y)

        # print("S1", model.score(np.column_stack([self.X_treated, np.ones(self.X_treated.shape[0])]), self.y_treated))
        # print("S2", model.score(np.column_stack([self.X_treated, np.zeros(self.X_treated.shape[0])]), self.y_treated))

        # ATT-> Predict on the treated group only
        y_predict_treated = model.predict(np.column_stack([self.X_treated, np.ones(self.X_treated.shape[0])]))
        y_predict_not_treated = model.predict(np.column_stack([self.X_treated, np.zeros(self.X_treated.shape[0])]))
        return np.mean(y_predict_treated - y_predict_not_treated)

    def t_learner_att(self):
        treated_model = GaussianProcessRegressor()
        control_model = GaussianProcessRegressor()
        # treated_model = LinearRegression()
        # control_model = LinearRegression()
        treated_model.fit(self.X_treated, self.y_treated)
        control_model.fit(self.X_control, self.y_control)

        pred_treated = treated_model.predict(self.X_treated)
        # r_square_graph(pred_treated, self.y_treated, 'GaussianProcessRegressor f(x,1) = y')
        pred_control = control_model.predict(self.X_treated)
        # r_square_graph(pred_control, self.y_treated, 'GaussianProcessRegressor f(x,0) = y')
        # print('R^2 of treated_model', treated_model.score(self.X_treated, self.y_treated))
        # print('R^2 of control_model', control_model.score(self.X_treated, self.y_treated))

        return np.mean(treated_model.predict(self.X_treated) - control_model.predict(self.X_treated))

    def matching_att(self):
        model = KNeighborsRegressor(algorithm='auto', metric='euclidean', n_neighbors=1)

        # Propensity Score Matching
        # control_propensity = self.propensity_score(self.X_control).reshape([-1, 1])
        # treated_propensity = self.propensity_score(self.X_treated).reshape([-1, 1])
        # model.fit(control_propensity, self.y_control)
        # y_predicted = model.predict(treated_propensity)

        # Fit the model to the control group only
        model.fit(self.X_control, self.y_control)

        # Predict for treated only
        y_predicted = model.predict(self.X_treated)

        return np.mean(self.y_treated - y_predicted)

    def return_all_att(self):
        att = [self.ipw_att(), self.s_learner_att(), self.t_learner_att(), self.matching_att()]
        return att

    # def print_histogram(self, col_idx):
    #     treat_plt = plt.hist(self.X_treated[:, col_idx], bins=100, label='Treated')
    #     control_plt = plt.hist(self.X_control[:, col_idx], bins=100, label='Control')
    #     plt.legend()
    #     plt.xlabel(f'{self.df.columns[col_idx]}')
    #     plt.ylabel('number of observations')
    #     plt.show()


def main():
    df_data1 = pd.read_csv('data1.csv', index_col=0)
    df_data2 = pd.read_csv('data2.csv', index_col=0)

    estimator1 = ATT_Estimator(df_data1)
    estimator2 = ATT_Estimator(df_data2)

    att1 = estimator1.return_all_att()
    att2 = estimator2.return_all_att()

    att_df = pd.DataFrame(zip(range(1, 6, 1), att1, att2), columns=['Type', 'data1', 'data2'])
    #print(att_df)
    att_df.to_csv('ATT_results.csv', index=False)

    propensity_df = pd.DataFrame(data=[estimator1.all_propensity_score, estimator2.all_propensity_score],
                                 index=['data1', 'data2'])
    #print(propensity_df)
    propensity_df.to_csv('models_propensity.csv', header=False)


if __name__ == '__main__':
    main()
