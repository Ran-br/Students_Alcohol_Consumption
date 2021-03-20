from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_curve, auc

g_count = 0
files = ['mat', 'por', 'comb']

def print_graphs(df):
    ####################################################################
    ######################   Correlation Matrix    ###################
    ####################################################################
    plt.figure(figsize=(10, 9))
    plt.title("Correlation Matrix")
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cbar=True, cmap="YlGnBu")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    ####################################################################
    ##########################   Ages    ###############################
    ####################################################################
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    sns.countplot(x="age", palette="mako", data=df, ax=ax1)
    plt.title("Age Histogram")

    ax2 = fig.add_subplot(122)
    sns.distplot(df["age"], color="green", norm_hist=True, ax=ax2)
    plt.title("Age Distribution")
    plt.show()

    # ####################################################################
    # ##########################   Grades    #############################
    # ####################################################################
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    sns.countplot(x="G3", palette="mako", data=df, ax=ax1)
    plt.title("Final Grade Histogram")

    ax2 = fig.add_subplot(122)
    sns.distplot(df["G3"], color="green", ax=ax2)
    plt.title("Final Grade Distribution")
    plt.show()

    # ####################################################################
    # ##########################   Walc / Dalc    ########################
    # ####################################################################
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    sns.countplot(x="Walc", palette="mako", data=df, ax=ax1)
    plt.title("Walc Histogram")

    ax2 = fig.add_subplot(122)
    sns.countplot(x="Dalc", palette="mako", data=df, ax=ax2)
    # sns.distplot(df["G3"], color="green", ax=ax2)
    plt.title("Dalc Histogram")
    plt.show()

    ####################################################################
    ########################   Walc + Dalc    ##########################
    ####################################################################
    df['Walc+Dalc'] = df['Walc'] + df['Dalc']
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    sns.countplot(x="Walc+Dalc", palette="mako", data=df, ax=ax1)
    plt.title("Walc+Dalc Histogram")

    ax2 = fig.add_subplot(122)
    sns.distplot(df["Walc+Dalc"], color="green", ax=ax2)
    plt.title("Walc+Dalc Distribution")
    plt.show()
    df = df.drop("Walc+Dalc", inplace=False, axis=1)

    global g_count

    ####################################################################
    ######################   Grades and Dalc+Walc    ###################
    ####################################################################

    df['Walc+Dalc'] = df['Walc'] + df['Dalc']
    groupedvaluesDalc_Walc = df.groupby('Walc+Dalc').mean().reset_index()

    groupedvalues_len = df.groupby('Walc+Dalc').size().reset_index(name='counts')
    g = sns.barplot(x='Walc+Dalc', y='G3', data=groupedvaluesDalc_Walc)

    pal = sns.color_palette("Purples_d", len(groupedvaluesDalc_Walc))
    rank = groupedvalues_len["counts"].argsort().argsort()
    g = sns.barplot(x='Walc+Dalc', y='G3', data=groupedvaluesDalc_Walc, palette=np.array(pal[::-1])[rank])

    for index, row, row_len in zip(range(0, np.max(df['Walc+Dalc']) + 1 - np.min(df['Walc+Dalc'])),
                                   groupedvaluesDalc_Walc.iterrows(),
                                   groupedvalues_len.iterrows()):
        g.text(index, 1, f'Count:\n{row_len[1].counts}', color='black', ha="center")

    g.set(xlabel='Walc + Dalc', ylabel='G3 Group Average')
    g.set_title("Final Grade Averages vs Total Alcohol Consumption")

    plt.show()
    df = df.drop("Walc+Dalc", inplace=False, axis=1)

    ####################################################################
    ############################## Box Plots ###########################
    ####################################################################
    df['Walc+Dalc'] = df['Walc'] + df['Dalc']

    fig = plt.figure(figsize=(9, 4))
    g = sns.boxplot(x="Walc+Dalc", y="G3", data=df, palette='mako')
    g.set(xlabel='Walc + Dalc', ylabel='G3')
    plt.title("Box Plot of Final Grades Distribution Over Alcohol Usage")
    plt.show()

    ave = sum(df.G3) / float(len(df))
    df['ave_line'] = ave
    df['average'] = ['above average' if i > ave else 'under average' for i in df.G3]
    g = sns.swarmplot(x='Walc+Dalc', y='G3', hue='average', hue_order=['above average', 'under average'], data=df,
                      palette='mako')
    g.set(xlabel='Walc + Dalc', ylabel='G3')
    g.set_title("Final Grade vs Total Alcohol Consumption")
    # replace labels
    for t, l in zip(g.legend().texts, ['Above average', 'Under average']): t.set_text(l)
    plt.show()

    df = df.drop(['ave_line', 'average'], inplace=False, axis=1)

    df = df.drop("Walc+Dalc", inplace=False, axis=1)

    ####################################################################
    #######################   Grades and Dalc    #######################
    ####################################################################
    #
    groupedvalues = df.groupby('Dalc').mean().reset_index()
    groupedvalues_len = df.groupby('Dalc').size().reset_index(name='counts')
    g = sns.barplot(x='Dalc', y='G3', data=groupedvalues)

    pal = sns.color_palette("Blues_d", len(groupedvalues))
    rank = groupedvalues_len["counts"].argsort().argsort()
    g = sns.barplot(x='Dalc', y='G3', data=groupedvalues, palette=np.array(pal[::-1])[rank])

    for index, row, row_len in zip(range(0, np.max(df.Dalc) + 1 - np.min(df.Dalc)),
                                   groupedvalues.iterrows(),
                                   groupedvalues_len.iterrows()):
        g.text(index, 1, f'Count:\n{row_len[1].counts}', color='black', ha="center")
    g.set(xlabel='Dalc', ylabel='G3 Group Average')
    g.set_title("Final Grade Averages vs Weekdays Alcohol Consumption")
    plt.show()

    ####################################################################
    #######################   Grades and Walc    #######################
    ####################################################################

    groupedvalues = df.groupby('Walc').mean().reset_index()
    groupedvalues_len = df.groupby('Walc').size().reset_index(name='counts')
    pal = sns.color_palette("Greens_d", len(groupedvalues))
    rank = groupedvalues_len["counts"].argsort().argsort()
    g = sns.barplot(x='Walc', y='G3', data=groupedvalues, palette=np.array(pal[::-1])[rank])
    g.set(xlabel='Walc', ylabel='G3 Group Average')
    g.set_title("Final Grade Averages vs Weekend Alcohol Consumption")

    for index, row, row_len in zip(range(0, np.max(df.Walc) + 1 - np.min(df.Walc)),
                                   groupedvalues.iterrows(),
                                   groupedvalues_len.iterrows()):
        g.text(index, 1, f'Count:\n{row_len[1].counts}', color='black', ha="center")
    plt.show()

    ####################################################################
    #######################   Grades and Ages    #######################
    ####################################################################
    groupedvalues = df.groupby('age').mean().reset_index()
    groupedvalues_len = df.groupby('age').size().reset_index(name='counts')
    g = sns.barplot(x='age', y='G3', data=groupedvalues)

    pal = sns.color_palette("Reds_d", len(groupedvalues))
    rank = groupedvalues_len["counts"].argsort().argsort()
    g = sns.barplot(x='age', y='G3', data=groupedvalues, palette=np.array(pal[::-1])[rank])

    for index, row, row_len in zip(range(0, np.max(df.age) + 1 - np.min(df.age)),
                                   groupedvalues.iterrows(),
                                   groupedvalues_len.iterrows()):
        g.text(index, 1, f'Count:\n{row_len[1].counts}', color='black', ha="center")
    plt.show()
    g_count += 1


def print_common_support_graphs(df):
    # Plot overlap (common support) of all features
    for column in df:
        fig, ax = plt.subplots()
        df.groupby('T')[column].plot(kind='hist', sharex=True, bins=30, alpha=0.75)
        ax.legend(["Control", "Treated"])
        plt.xlabel(f'{column}')
        plt.ylabel('number of observations')
        plt.show()

def pre_process(df):
    # Print all the graphs for the raw data
    print_graphs(deepcopy(df))

    # Find and split all columns to categorical and numerical
    categorical_feature_mask = df.dtypes == object
    categorical_features = df.columns[categorical_feature_mask].tolist()
    numeric_feature_mask = df.dtypes != object
    numeric_features = df.columns[numeric_feature_mask].tolist()

    df = df[numeric_features].join(pd.get_dummies(df[categorical_features]))

    # Define the treatment as Dalc + Walc
    treatment = ['Dalc', 'Walc']
    df['T'] = np.where(df[treatment[0]]+df[treatment[1]] >= 6, 1, -1)
    df['T'].where(df[treatment[0]]+df[treatment[1]] > 2, 0, inplace=True)  # Careful here, pandas 'where' does the opposite of np.where
    df = df[df['T'] != -1]
    df.drop(treatment, inplace=True, axis=1)

    # Define Treated and Non-treated separation
    # treatment = 'Dalc'
    # df['T'] = np.where(df[treatment] >= 3, 1, -1)
    # df['T'].where(df[treatment] > 1, 0, inplace=True) # Careful here, pandas 'where' does the opposite of np.where
    # df = df[df['T'] != -1]
    # df.drop([treatment], inplace=True, axis=1)

    # Visualize the overlap between the treated group and the control
    print_common_support_graphs(deepcopy(df))

    return df


''' Split the DataFrame to smaller units of data which are easier to work with '''
def split_data(df):

    # Normilize the data and store separate it for easier work a head
    if 'propensity' in df.columns:
        df = df.drop(['propensity'], inplace=False, axis=1)

    X = MinMaxScaler().fit_transform(df.drop(['T', 'G1', 'G2', 'G3'], axis=1))
    T, y = df['T'], df['G3']
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


class AverageTreatmentEstimator:
    def __init__(self, data):
        self.df = pre_process(data)
        (self.X,
         self.X_treated,
         self.X_control,
         self.y,
         self.y_treated,
         self.y_control,
         self.T) = split_data(self.df)
        self.prop_model = self.calc_propensity()
        self.all_propensity_score = self.propensity_score(self.X)

        # Trim common support (overlap)
        (self.X_trimmed,
         self.X_treated_trimmed,
         self.X_control_trimmed,
         self.y_trimmed,
         self.y_treated_trimmed,
         self.y_control_trimmed,
         self.T_trimmed) = split_data(self.trim_common_support(self.df))

        print("Num after trim ", len(self.X_trimmed))

        ###############################################################
        ################ Before Trimming Common Support   ##############
        ###############################################################
        # fig, ax = plt.subplots()
        # self.df.groupby('T')["propensity"].plot(kind='hist', sharex=True, bins=10, alpha=0.5)
        # ax.set_xlim([0, 1])
        # ax.set_title("Original")
        # ax.legend(["Control", "Treated"])
        # plt.xlabel('propensity')
        # plt.ylabel('number of observations')
        # #plt.ylim([0, 100])
        # plt.show()

        self.df = self.trim_common_support(self.df)
        (self.X,
         self.X_treated,
         self.X_control,
         self.y,
         self.y_treated,
         self.y_control,
         self.T) = split_data(self.df)

        ###############################################################
        ################ After Trimming Common Support   ##############
        ###############################################################
        # fig, ax = plt.subplots()
        # ax.set_xlim([0, 1])
        # self.df.groupby('T')["propensity"].plot(kind='hist', sharex=True, bins=10, alpha=0.5)
        # ax.set_title("After Trimming")
        # ax.legend(["Control", "Treated"])
        # plt.xlabel('propensity')
        # plt.ylabel('number of observations')
        # #plt.ylim([0, 100])
        # plt.show()

    def calc_propensity(self):
        model = LogisticRegression() # We can add C=1e6 to cancel default sklearn regularization
        model.fit(self.X, self.T)

        # # Print ROC to check propensity model
        # t_pred = model.predict(self.X)
        # print('Accuracy', model.score(self.X, self.T))
        # print('Brier', np.mean((t_pred - self.y) ** 2))
        # fpr, tpr, _ = roc_curve(self.T, t_pred)
        # plt.plot(fpr, tpr, linestyle='dotted', label=f'ROC curve (AUC = {auc(fpr, tpr)})')
        # plt.plot([0, 1], [0, 1], linestyle='dashed', label='Logistic Regression')
        # plt.legend()
        # plt.show()

        return model

    def propensity_score(self, X_values):
        return self.prop_model.predict_proba(X_values)[:, -1]

    def trim_common_support(self, data):
        data['propensity'] = self.propensity_score(self.X)
        # Save results to a file for easier read
        #data.to_csv("with_propensity.csv", index=False)

        group_min_max = (data.groupby('T').propensity.agg(min_propensity=np.min, max_propensity=np.max))

        # Compute boundaries of common support between the two propensity score distributions
        min_common_support = np.max(group_min_max.min_propensity)
        max_common_support = np.min(group_min_max.max_propensity)

        common_support_filter = (data.propensity >= min_common_support) & (data.propensity <= max_common_support)

        return data[common_support_filter]

    def ipw_att(self):
        # Calc the propensity of the control group (not treated)
        e_x = self.propensity_score(self.X_control)
        weighted_e_x = e_x / (1 - e_x)

        # Use the formula from class to calculate ATT with IPW
        att = (np.sum(self.y_treated) / len(self.y_treated) -
               np.sum(self.y_control * weighted_e_x) / np.sum(weighted_e_x))
        return att

    def ipw_ate(self):
        # Calc the propensity of the control group (not treated)
        e_x = self.propensity_score(self.X)

        # Use the formula from class to calculate ATE with IPW
        ate = np.mean((self.T * self.y)/e_x) - np.mean(((1 - self.T) * self.y) / (1 - e_x))
        return ate

    def s_learner_att(self, model=GaussianProcessRegressor()):
        model.fit(np.column_stack([self.X, self.T]), self.y)

        # print("S1", model.score(np.column_stack([self.X_treated, np.ones(self.X_treated.shape[0])]), self.y_treated))
        # print("S2", model.score(np.column_stack([self.X_treated, np.zeros(self.X_treated.shape[0])]), self.y_treated))

        # ATT-> Predict on the treated group only
        y_predict_treated = model.predict(np.column_stack([self.X_treated, np.ones(self.X_treated.shape[0])]))
        y_predict_not_treated = model.predict(np.column_stack([self.X_treated, np.zeros(self.X_treated.shape[0])]))
        return np.mean(y_predict_treated - y_predict_not_treated)

    def s_learner_ate(self, model=GaussianProcessRegressor()):
        model.fit(np.column_stack([self.X, self.T]), self.y)

        ####################################################################
        #######################   Coefficiants    ##########################
        ####################################################################
        # cdf = pd.DataFrame(model.coef_, self.df.drop(['propensity', 'G1', 'G2', 'G3'], axis=1).columns, columns=['Coefficients'])
        # #print(sorted(cdf.to_dict().values(), key=lambda x: np.absolute(x)))
        # print()
        # print(cdf.assign(sortval=np.abs(cdf.Coefficients)).sort_values('sortval', ascending=False).drop('sortval', 1))
        #
        # cdf.plot(kind='barh', figsize=(9, 7))
        # plt.title('Ridge model, small regularization')
        # plt.axvline(x=0, color='.5')
        # plt.subplots_adjust(left=.3)
        # plt.show()

        # print("S1", model.score(np.column_stack([self.X_treated, np.ones(self.X_treated.shape[0])]), self.y_treated))
        # print("S2", model.score(np.column_stack([self.X_treated, np.zeros(self.X_treated.shape[0])]), self.y_treated))

        # ATE
        y_predict_treated = model.predict(np.column_stack([self.X, np.ones(self.X.shape[0])]))
        y_predict_not_treated = model.predict(np.column_stack([self.X, np.zeros(self.X.shape[0])]))

        print('\nS_Learner RMSE: pred_treated %.2f' % mean_squared_error(self.y, y_predict_treated) ** 0.5)
        print('S_Learner RMSE: pred_control %.2f' % mean_squared_error(self.y, y_predict_not_treated) ** 0.5)
        print('S_Learner Real std %.2f' % np.std(self.y_treated))
        print('S_Learner Real std %.2f' % np.std(self.y_control))

        return np.mean(y_predict_treated - y_predict_not_treated)

    def t_learner_ate(self):
        treated_model = GaussianProcessRegressor()
        control_model = GaussianProcessRegressor()

        # treated_model = TweedieRegressor(power=1.9, alpha=1e10, max_iter=10000)
        # control_model = TweedieRegressor(power=1.9, alpha=1e10, max_iter=10000)
        # treated_model = Ridge(alpha=1e10)
        # control_model = Ridge(alpha=1e10)

        treated_model.fit(self.X_treated, self.y_treated)
        control_model.fit(self.X_control, self.y_control)

        pred_treated = treated_model.predict(self.X_treated)
        #r_square_graph(pred_treated, self.y_treated, 'GaussianProcessRegressor f(x,1) = y')
        pred_control = control_model.predict(self.X_control)
        #r_square_graph(pred_control, self.y_control, 'GaussianProcessRegressor f(x,0) = y')
        #print('t_learner R^2 of treated_model', treated_model.score(self.X_treated, self.y_treated))
        #print('t_learner R^2 of control_model', control_model.score(self.X_control, self.y_control))

        print('\nT_Learner RMSE: pred_treated', mean_squared_error(self.y_treated, pred_treated)**0.5)
        print('Real std ', np.std(self.y_treated))

        print('\nT_Learner RMSE: pred_control', mean_squared_error(self.y_control, pred_control) ** 0.5)
        print('T_ Learner Real std ', np.std(self.y_control))

        return np.mean(treated_model.predict(self.X) - control_model.predict(self.X))

    def matching_att(self):
        model = KNeighborsRegressor(algorithm='auto', metric='euclidean', n_neighbors=1)

        # Propensity Score Matching
        # control_propensity = self.propensity_score(self.X_control).reshape([-1, 1])
        # treated_propensity = self.propensity_score(self.X_treated).reshape([-1, 1])
        # model.fit(control_propensity, self.y_control)
        # y_predicted = model.predict(treated_propensity)

        # Train the model
        model.fit(self.X_control, self.y_control)

        # Predict for treated only
        y_predicted_treated = model.predict(self.X_treated)

        treated_mean = np.mean(self.y_treated - y_predicted_treated)

        return treated_mean

    def matching_ate(self):
        model_control = KNeighborsRegressor(algorithm='brute',
                                            metric='mahalanobis',
                                            metric_params={'VI': np.cov(self.X_treated)}, n_neighbors=3)
        model_treatment = KNeighborsRegressor(algorithm='brute',
                                              metric='mahalanobis',
                                              metric_params={'VI': np.cov(self.X_control)}, n_neighbors=3)
        # model_control = KNeighborsRegressor(algorithm='auto', metric='euclidean', n_neighbors=1)
        # model_treatment = KNeighborsRegressor(algorithm='auto', metric='euclidean', n_neighbors=1)

        # Train the model
        model_control.fit(self.X_control, self.y_control)
        model_treatment.fit(self.X_treated, self.y_treated)

        # Predict for treated only
        y_predicted_treated = model_control.predict(self.X_treated)
        y_predicted_control = model_treatment.predict(self.X_control)

        print('\nMatching RMSE: pred_treated %.2f' % mean_squared_error(self.y_treated, y_predicted_treated) ** 0.5)
        print('Matching RMSE: pred_control %.2f' % mean_squared_error(self.y_control, y_predicted_control) ** 0.5)
        print('Matching Real std %.2f' % np.std(self.y_treated))
        print('Matching Real std %.2f' % np.std(self.y_control))

        return np.mean((self.y_treated - y_predicted_treated).append(y_predicted_control - self.y_control))

        ####################################################################################
        ##################### Try propensity score matching for ATE ########################
        ####################################################################################
        # control_propensity = self.propensity_score(self.X_control).reshape([-1, 1])
        # treated_propensity = self.propensity_score(self.X_treated).reshape([-1, 1])
        # # Train the model
        # model_control.fit(control_propensity, self.y_control)
        # model_treatment.fit(treated_propensity, self.y_treated)
        #
        # # Predict for treated only
        # y_predicted_treated = model_control.predict(treated_propensity)
        # y_predicted_control = model_treatment.predict(control_propensity)
        #
        # # return np.mean(self.y_treated - y_predicted)
        # treated_mean = np.mean(self.y_treated - y_predicted_treated)
        # control_mean = np.mean(y_predicted_control - self.y_control)
        # return np.mean((self.y_treated - y_predicted_treated).append(y_predicted_control - self.y_control))

    # Use doubly robust estimator to try and decrease the possible error of estimation
    def doubly_robust(self):
        ps = self.propensity_score(self.X)

        mu0 = GaussianProcessRegressor().fit(self.X_control, self.y_control).predict(self.X)
        mu1 = GaussianProcessRegressor().fit(self.X_treated, self.y_treated).predict(self.X)

        g1 = mu1 + (self.T / ps) * (self.y - mu1)
        g0 = mu0 + ((1 - self.T) / (1 - ps)) * (self.y - mu0)

        # return (np.mean(self.T * (self.y - mu1) / ps + mu1) -
        #         np.mean((1 - self.T) * (self.y - mu0) / (1 - ps) + mu0))
        return np.mean(g1 - g0)

    def return_all_att(self):
        att = [self.ipw_att(), self.s_learner_att(), self.matching_att()]
        return att + [np.average(att)]

    def return_all_ate(self):
        ate = [self.ipw_ate(), self.s_learner_ate(), self.t_learner_ate(), self.matching_ate(), self.doubly_robust()]
        #ate_3_methods = [self.ipw_ate(), self.s_learner_ate(), self.matching_ate()]
        #return ate + [np.average(ate_3_methods), np.std(ate_3_methods)]
        return ate + [np.average(ate), np.std(ate)]

    def print_histogram(self, col_idx):
        treat_plt = plt.hist(self.X_treated[:, col_idx], bins=20, label='Treated')
        control_plt = plt.hist(self.X_control[:, col_idx], bins=20, label='Control')
        plt.legend()
        plt.xlabel(f'{self.df.columns[col_idx]}')
        plt.ylabel('number of observations')
        plt.show()

    def save_df(self):
        # Save results to a file for easier read
        self.df.to_csv("After_PP1.csv", index=False)


def main():
    df_mat = pd.read_csv('student-mat.csv')
    df_por = pd.read_csv('student-por.csv')

    df_combined = pd.concat([df_mat, df_por], ignore_index=True)

    # # Check for unique students
    # # Select all duplicate rows based on multiple column names in list
    # check_dupe = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]
    # df_no_dupe2 = df_combined[df_combined.duplicated(check_dupe)].reset_index()
    #
    # our_check_dupe = list(df_combined.drop(['G1', 'G2', 'G3', 'failures', 'absences'], axis=1).columns)
    # print("Columns to check for dupes:", [x for x in our_check_dupe if x not in check_dupe])
    # print('Mat: ', len(df_mat))
    # print('Por: ', len(df_por))
    # print('Combined: ', len(df_combined))
    # print('Dupes: ', len(df_no_dupe2))
    # print('Unique: ', len(df_combined)-len(df_no_dupe2))
    #
    # # Print final grades statistics
    # print("Mean grade:", np.mean(df_combined['G3']))
    # print("Median grade:", np.median(df_combined['G3']))
    # print("std grade:", np.std(df_combined['G3']))

    estimator1 = AverageTreatmentEstimator(df_mat)
    estimator2 = AverageTreatmentEstimator(df_por)
    estimator3 = AverageTreatmentEstimator(df_combined)

    # estimator1.save_df()
    # estimator2.save_df()
    # estimator3.save_df()

    att1 = estimator1.return_all_ate()
    att2 = estimator2.return_all_ate()
    att3 = estimator3.return_all_ate()

    methods = ['IPW', 'S_Learner', 'T_Learner', 'Matching', 'Doubly Robust', 'Average', 'STD']
    att_df = pd.DataFrame(zip(methods, att1, att2, att3), columns=['Type', 'mat', 'por', 'combined'])
    print(att_df.to_string(float_format=lambda x: '%.3f' % x))

if __name__ == '__main__':
    main()
