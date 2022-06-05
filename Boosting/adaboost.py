import numpy as np
from tqdm import tqdm
import pandas as pd
import time

#  binary classification of AdaBoost  #
#  library: numpy tqdm pandas         #


class AdaBoost:

    def __init__(self, n_estimators=10):
        """
        :param n_estimators: nums of basic classifier
        """

        # nums of basic estimators
        self.M = n_estimators

        # save estimators (list)
        self.estimators = []

        # initialize the weights of training data
        self.weights = None

    def divide_data(self, X, y, feature_div, rule, thresh):
        """
        divide the data and compute the error
        :param X:
        :param y: label of data
        :param feature_div: feature of data to divide
        :param relu: divide rule
        :param thresh: value of thresh to divide
        :return: predict result, error
        """

        # get divide feature value of X
        X_feature = X[:, feature_div]

        error = 0
        predict = []

        # iterate all the data
        for i in range(X.shape[0]):
            if rule == 1:
                pred = 1 if X_feature[i] >= thresh else -1
                predict.append(pred)
                error += 0 if pred == y[i] else self.weights[i]
            elif rule == -1:
                pred = 1 if X_feature[i] < thresh else -1
                predict.append(pred)
                error += 0 if pred == y[i] else self.weights[i]

        return predict, error


    def create_estimator(self, X, y):
        """
        create an estimator for new layer
        :return: an new estimator
        """
        estimator = {}

        # define the thresh of each feature to classify the data
        # use median value of each feature to be the thresh
        thresh = np.median(X, axis=0)

        error = np.inf

        # find the feature and method to divide the data
        # rule: 1 - put the data which larger than thresh on positive
        # rule: -1 - put the data which less than thresh on positive
        for fea_div in range(X.shape[1]):
            for rule in [-1, 1]:
                # divide the data
                predict, error_curr = self.divide_data(X, y, fea_div, rule, thresh[fea_div])

                if error_curr > error: continue
                error = error_curr

                # update the divide method
                estimator['feature_divide'] = fea_div
                estimator['thresh_divide'] = thresh[fea_div]
                estimator['rule_divide'] = rule
                estimator['predict'] = np.array(predict)
                estimator['error'] = error_curr

        return estimator


    def fit(self, X, y):
        """
        to fit the model
        :param X: feature of training data
        :param y: label of training data
        """
        # weights of data
        self.weights = np.ones(X.shape[0]) / X.shape[0]

        for _ in tqdm(range(self.M), colour='blue', desc='TRAIN'):

            estimator = self.create_estimator(X, y)

            # compute the coefficient of current estimator
            alpha_m = np.log((1 - estimator['error']) / (estimator['error'] + 1e-4)) / 2
            estimator['alpha'] = alpha_m

            # normalize factor
            norm_factor = np.sum(self.weights * np.exp(-alpha_m * y * estimator['predict']))

            # update the weights of data
            self.weights = self.weights * np.exp(-alpha_m * y * estimator['predict']) / norm_factor

            self.estimators.append(estimator)

    def predict(self, X):
        """
        predict labels of X
        :param X: data to predict
        :return: result of predict
        """
        predict = []

        for i in range(X.shape[0]):
            x = X[i]; result = 0

            # use all the basic estimator to weighted voting
            for estimator in self.estimators:

                feature_div = estimator['feature_divide']
                thresh_div = estimator['thresh_divide']
                rule = estimator['rule_divide']
                alpha = estimator['alpha']

                if rule == 1:
                    pred = 1 if x[feature_div] >= thresh_div else -1
                else:
                    pred = 1 if x[feature_div] < thresh_div else -1
                result += pred * alpha

            predict.append(np.sign(result))

        return predict



if __name__ == '__main__':

    # load data(iris) and remove the first column (id)
    # with no feature processing
    data = np.array(pd.read_csv('../data/iris.txt', header=None))[:, 1:]

    # change three-way classification to two-way
    data[:, -1][data[:, -1] == 2] = -1
    data[:, -1][data[:, -1] != -1] = 1

    # random disturb
    np.random.shuffle(data)

    # split train dataset and test dataset
    X_train, y_train = data[:120, :-1], data[:120, -1]
    X_test, y_test = data[120:, :-1], data[120:, -1]

    time1 = time.time()

    # ---- create and fit the model ---- #
    model = AdaBoost(n_estimators=500)
    model.fit(X_train, y_train)

    print('Spend ', time.time() - time1, 's to fitting')
    time2 = time.time()

    # ---- predict ---- #
    predict = model.predict(X_test)
    print('Spend ', time.time() - time2, 's to predicting')

    print('Accuracy: ', np.sum(predict == y_test) / len(predict) * 100, '%')