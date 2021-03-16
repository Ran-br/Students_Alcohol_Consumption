import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor
from sklearn.linear_model import Ridge

g_count = 0
files = ['mat', 'por', 'comb']

def pre_process(df):
    # Convert all 'yes' 'no' values to 0 and 1
    #print(df.to_string())
    df.replace('yes', 1, inplace=True)
    df.replace('no', 0, inplace=True)

    # Find and split all columns to categorical and numerical
    categorical_feature_mask = df.dtypes == object
    categorical_features = df.columns[categorical_feature_mask].tolist()
    numeric_feature_mask = df.dtypes != object
    numeric_features = df.columns[numeric_feature_mask].tolist()

    df = df[numeric_features].join(pd.get_dummies(df[categorical_features]))

    # Define Treated and Non-treated separation
    # treatment = 'Dalc'
    # df['T'] = np.where(df[treatment] >= 2, 1, -1)
    # df['T'].where(df[treatment] > 1, 0, inplace=True) # Careful here, pandas 'where' does the opposite of np.where
    # df = df[df['T'] != -1]
    # df.drop([treatment], inplace=True, axis=1)

    global g_count

    # # Average grade
    # list1 = []
    # for i in range(1, 6):
    #     list1.append(sum(df[df.Dalc == i].G3) / float(len(df[df.Dalc == i])))
    # ax = sns.barplot(x=[1, 2, 3, 4, 5], y=list1)
    # ax.set_title(files[g_count])
    # plt.ylabel('Average Grades of students')
    # plt.xlabel('Weekly alcohol consumption')
    # plt.show()
    #
    # # Average grade
    # list2 = []
    # for i in range(2, 11):
    #     list2.append(sum(df[df.Dalc + df.Walc == i].G3) / float(len(df[df.Dalc + df.Walc == i])))
    # ax = sns.barplot(x=[2, 3, 4, 5, 6, 7, 8, 9, 10], y=list2)
    # ax.set_title(files[g_count])
    # plt.ylabel('Average Grades of students')
    # plt.xlabel('Total alcohol consumption')
    # plt.show()

                # # Average grade
                # ages_range = range(np.min(df.age), np.max(df.age)+1)
                # age_avg = []
                # age_count = []
                # for i in ages_range:
                #     age_avg.append(sum(df[df.age == i].G3) / float(len(df[df.age == i])))
                #     age_count.append(len(df[df.age == i]))
                # ax = sns.barplot(x=list(ages_range), y=age_avg)
                # for age, count, avg in zip(ages_range, age_count, age_avg):
                #     ax.text(age, avg, count, color='black', ha="center")
    #
    # ax.set_title(files[g_count])
    # plt.ylabel('Average Grades of students')
    # plt.xlabel('Age')
    # plt.show()


    ####################################################################
    df['Walc+Dalc'] = df['Walc']+df['Dalc']
    groupedvaluesDalc_Walc = df.groupby('Walc+Dalc').mean().reset_index()

    groupedvalues_len = df.groupby('Walc+Dalc').size().reset_index(name='counts')
    g = sns.barplot(x='Walc+Dalc', y='G3', data=groupedvaluesDalc_Walc)

    pal = sns.color_palette("Reds_d", len(groupedvaluesDalc_Walc))
    rank = groupedvalues_len["counts"].argsort().argsort()
    g = sns.barplot(x='Walc+Dalc', y='G3', data=groupedvaluesDalc_Walc, palette=np.array(pal[::-1])[rank])

    for index, row, row_len in zip(range(0, np.max(df['Walc+Dalc']) + 1 - np.min(df['Walc+Dalc'])),
                                   groupedvaluesDalc_Walc.iterrows(),
                                   groupedvalues_len.iterrows()):
        g.text(index, 2, f'Count:\n{row_len[1].counts}', color='black', ha="center")
    plt.show()
    df = df.drop("Walc+Dalc", inplace=False, axis=1)

    ####################################################################


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
    plt.show()
    ####################################################################

    groupedvalues = df.groupby('Walc').mean().reset_index()
    groupedvalues_len = df.groupby('Walc').size().reset_index(name='counts')
    g = sns.barplot(x='Walc', y='G3', data=groupedvalues)

    pal = sns.color_palette("Oranges_d", len(groupedvalues))
    rank = groupedvalues_len["counts"].argsort().argsort()
    g = sns.barplot(x='Walc', y='G3', data=groupedvalues, palette=np.array(pal[::-1])[rank])
    g.set(xlabel='Guyhomo', ylabel='common ylabel')
    g.set_title("guygever")

    for index, row, row_len in zip(range(0, np.max(df.Walc) + 1 - np.min(df.Walc)),
                                   groupedvalues.iterrows(),
                                   groupedvalues_len.iterrows()):
        g.text(index, 1, f'Count:\n{row_len[1].counts}', color='black', ha="center")
    plt.show()
    ####################################################################


    groupedvalues = df.groupby('age').mean().reset_index()
    groupedvalues_len = df.groupby('age').size().reset_index(name='counts')
    g = sns.barplot(x='age', y='G3', data=groupedvalues)

    pal = sns.color_palette("Greens_d", len(groupedvalues))
    rank = groupedvalues_len["counts"].argsort().argsort()
    g = sns.barplot(x='age', y='G3', data=groupedvalues, palette=np.array(pal[::-1])[rank])

    for index, row, row_len in zip(range(0, np.max(df.age)+1 - np.min(df.age)),
                          groupedvalues.iterrows(),
                          groupedvalues_len.iterrows()):
        g.text(index, 1, f'Count:\n{row_len[1].counts}', color='black', ha="center")
    plt.show()




    g_count += 1
    # # Average grade
    # list = []
    # for i in range(1, 6):
    #     list.append(sum(df[df.Walc == i].G3) / float(len(df[df.Walc == i])))
    # ax = sns.barplot(x=[1, 2, 3, 4, 5], y=list)
    # plt.ylabel('Average Grades of students')
    # plt.xlabel('Weekly alcohol consumption')
    # plt.show()

    treatment = ['Dalc', 'Walc']
    df['T'] = np.where(df[treatment[0]]+df[treatment[1]] >= 6, 1, -1)
    df['T'].where(df[treatment[0]]+df[treatment[1]] > 2, 0, inplace=True)  # Careful here, pandas 'where' does the opposite of np.where
    df = df[df['T'] != -1]
    df.drop(treatment, inplace=True, axis=1)

    # Plot overlap (common support) of all features
    # for column in df:
    #     fig, ax = plt.subplots()
    #     df.groupby('T')[column].plot(kind='hist', sharex=True, bins=30, alpha=0.75)
    #     ax.legend(["Control", "Treated"])
    #     plt.xlabel(f'{column}')
    #     plt.ylabel('number of observations')
    #     plt.show()

    # Save results to a file for easier read
    # df.to_csv("After_PP.csv", index=False)

    return df

def split_data(df):

    # Normilize the data and store separate it for easier work a head
    if 'propensity' in df.columns:
        df = df.drop(['propensity'], inplace=False, axis=1)

    X = MinMaxScaler().fit_transform(df.drop(['T', 'G1', 'G2', 'G3'], axis=1))
    T, y = df['T'], df['G3']
    X_treated, X_control = X[df['T'] == 1], X[df['T'] == 0]
    y_treated, y_control = y[df['T'] == 1], y[df['T'] == 0]

    return X, X_treated, X_control, y, y_treated, y_control, T

    # X = MinMaxScaler().fit_transform(df.drop(['T', 'Y'], axis=1))
    # T, y = df['T'], df['Y']
    # X_treated, X_control = X[df['T'] == 1], X[df['T'] == 0]
    # y_treated, y_control = y[df['T'] == 1], y[df['T'] == 0]
    #
    # return X, X_treated, X_control, y, y_treated, y_control, T


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

        # # Plot avg scores
        # # for column in df:
        # fig, ax = plt.subplots()
        # self.df.groupby('age')["G3"].plot(kind='hist', sharex=False, bins=30, alpha=0.5)
        # ax.legend()
        # ax.set_title("Original")
        # plt.ylabel('number of observations')
        # plt.show()



        # fig, ax = plt.subplots()
        # self.df.groupby('T')["propensity"].plot(kind='hist', sharex=True, bins=30, alpha=0.5)
        # ax.set_xlim([0, 1])
        # ax.set_title("Original")
        # ax.legend(["Control", "Treated"])
        # plt.xlabel('propensity')
        # plt.ylabel('number of observations')
        # plt.show()
        #
        # fig, ax = plt.subplots()
        # ax.set_xlim([0, 1])
        # self.trim_common_support(self.df).groupby('T')["propensity"].plot(kind='hist', sharex=True, bins=30, alpha=0.5)
        # ax.set_title("After Trimming")
        # ax.legend(["Control", "Treated"])
        # plt.xlabel('propensity')
        # plt.ylabel('number of observations')
        # plt.show()

        (self.X,
         self.X_treated,
         self.X_control,
         self.y,
         self.y_treated,
         self.y_control,
         self.T) = split_data(self.trim_common_support(self.df))

        # temp = np.column_stack([self.X, self.T, self.y])
        # df_temp = pd.DataFrame(temp)

        #df_temp[7].hist(by=df_temp[46])

        #self.df['famsup'].hist(by=self.df['paid'])

        # for i in range(40):
        #     self.print_histogram(i)
        #df_temp.groupby(46)[3].plot(kind='hist', sharex=True, range=(0, 40000), bins=30, alpha=0.75)
        #self.df.groupby('paid')['famsup'].plot(kind='hist', sharex=True, range=(0, 40000), bins=30, alpha=0.75)

    def calc_propensity(self):
        model = LogisticRegression() # We can add C=1e6 to cancel default sklearn regularization
        model.fit(self.X, self.T)
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

    def s_learner_ate(self, model=Ridge()):
        model.fit(np.column_stack([self.X, self.T]), self.y)

        cdf = pd.DataFrame(model.coef_, self.df.drop(['propensity', 'G1', 'G2', 'G3'], axis=1).columns, columns=['Coefficients'])
        #print(sorted(cdf.to_dict().values(), key=lambda x: np.absolute(x)))
        print()
        print(cdf.assign(sortval=np.abs(cdf.Coefficients)).sort_values('sortval', ascending=False).drop('sortval', 1))

        cdf.plot(kind='barh', figsize=(9, 7))
        plt.title('Ridge model, small regularization')
        plt.axvline(x=0, color='.5')
        plt.subplots_adjust(left=.3)
        plt.show()
        # print("S1", model.score(np.column_stack([self.X_treated, np.ones(self.X_treated.shape[0])]), self.y_treated))
        # print("S2", model.score(np.column_stack([self.X_treated, np.zeros(self.X_treated.shape[0])]), self.y_treated))

        # ATE
        y_predict_treated = model.predict(np.column_stack([self.X, np.ones(self.X.shape[0])]))
        y_predict_not_treated = model.predict(np.column_stack([self.X, np.zeros(self.X.shape[0])]))
        return np.mean(y_predict_treated - y_predict_not_treated)

    def t_learner_ate(self):
        treated_model = GaussianProcessRegressor()
        control_model = GaussianProcessRegressor()
        # treated_model = LinearRegression()
        # control_model = LinearRegression()
        treated_model.fit(self.X_treated, self.y_treated)
        control_model.fit(self.X_control, self.y_control)

        pred_treated = treated_model.predict(self.X_treated)
        #r_square_graph(pred_treated, self.y_treated, 'GaussianProcessRegressor f(x,1) = y')
        pred_control = control_model.predict(self.X_control)
        #r_square_graph(pred_control, self.y_control, 'GaussianProcessRegressor f(x,0) = y')
        #print('t_learner R^2 of treated_model', treated_model.score(self.X_treated, self.y_treated))
        #print('t_learner R^2 of control_model', control_model.score(self.X_control, self.y_control))

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
        y_predicted = model.predict(self.X_treated)

        return np.mean(self.y_treated - y_predicted)

    # Use doubly robust estimator to try and decrease the possible error of estimation
    def doubly_robust(self):
        ps = self.propensity_score(self.X)
        mu0 = GaussianProcessRegressor().fit(self.X_control, self.y_control).predict(self.X)
        mu1 = GaussianProcessRegressor().fit(self.X_treated, self.y_treated).predict(self.X)
        return (np.mean(self.T * (self.y - mu1) / ps + mu1) -
                np.mean((1 - self.T) * (self.y - mu0) / (1 - ps) + mu0))

    def return_all_att(self):
        att = [self.ipw_att(), self.s_learner_att(), self.t_learner_ate(), self.matching_att(), self.doubly_robust()]
        return att + [np.average(att)]

    def return_all_ate(self):
        att = [self.ipw_ate(), self.s_learner_ate(), self.t_learner_ate(), self.matching_att(), self.doubly_robust()]
        return att + [np.average(att)]

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

    print(len(df_mat))
    print(len(df_por))
    print(len(df_combined))

    # years = [2014, 2014, 2014, 2015, 2015, 2015, 2015]
    # vehicle_types = ['Truck', 'Truck', 'Car', 'Bike', 'Truck', 'Bike', 'Car']
    # companies = ["Mercedez", "Tesla", "Tesla", "Yamaha", "Tesla", "BMW", "Ford"]
    #
    # df = pd.DataFrame({'year': years,
    #                    'vehicle_type': vehicle_types,
    #                    'company': companies
    #                    })

    #df.groupby('paid')['famsup'].value_counts().unstack().plot.bar()
   # plt.show()
    # grouped = df_mat.groupby('paid')
    # plt.figure()
    #
    #
    # for group in grouped:
    #     # plt.hist(group[1].famsup)
    #     treat_plt = plt.hist(group[1].famsup, bins=20, label='Treated')
    #     #control_plt = plt.hist(group[1].famsup, bins=20, label='Control')
    # plt.legend()
    # plt.xlabel(f'{grouped.columns["famsup"]}')
    # plt.ylabel('number of observations')
    #
    # plt.show()

    #X, X_treated, X_control, y, y_treated, y_control, T = split_and_preprocess(df_mat)

    estimator1 = AverageTreatmentEstimator(df_mat)
    estimator2 = AverageTreatmentEstimator(df_por)
    estimator3 = AverageTreatmentEstimator(df_combined)

    estimator1.save_df()
    estimator2.save_df()
    estimator3.save_df()

    att1 = estimator1.return_all_ate()
    att2 = estimator2.return_all_ate()
    att3 = estimator3.return_all_ate()


    att_df = pd.DataFrame(zip(range(1, 7, 1), att1, att2, att3), columns=['Type', 'mat', 'por', 'combined'])
    print(att_df)
    # att_df.to_csv('ATT_results.csv', index=False)
    #
    # propensity_df = pd.DataFrame(data=[estimator1.all_propensity_score, estimator2.all_propensity_score],
    #                              index=['data1', 'data2'])
    # print(propensity_df)
    # propensity_df.to_csv('models_propensity.csv', header=False)


if __name__ == '__main__':
    main()
