import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA

import keras


class LIMESimpleModel:
    def __init__(self, cluster_num, cluster_method=KMeans, random_state=None):
        self.random_state = check_random_state(random_state)
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method(
            n_clusters=cluster_num, random_state=self.random_state)
        self.models = []

    def fit(self, X, y, predict_fn, labels_num):
        self.cluster_labels = self.cluster_method.fit_predict(X)
        self.labels_num = labels_num

        for i in range(self.cluster_num):
            inds = np.where(self.cluster_labels == i)

            simplified_models = LinearRegression()
            simplified_models.fit(X[inds], y[inds])

            coef_ = simplified_models.coef_.T
            intercept_ = simplified_models.intercept_

            self.models.append((coef_, intercept_))

    def predict(self, x):

        cluster_result = self.cluster_method.predict(x)
        prediction_result = np.zeros(x.shape[0])

        for i in range(self.cluster_num):
            inds = np.where(cluster_result == i)
            if not len(inds[0]):
                continue
            predict_values = np.dot(x[inds],
                                    self.models[i][0]) + self.models[i][1]
            prediction_result[inds] = np.argmax(predict_values, axis=1)

        return prediction_result

    def predict_reg(self, x):

        cluster_result = self.cluster_method.predict(x)
        predict_values = np.zeros((x.shape[0], self.labels_num))

        for i in range(self.cluster_num):
            inds = np.where(cluster_result == i)
            if not len(inds[0]):
                continue
            predict_values[inds] = np.dot(
                x[inds], self.models[i][0]) + self.models[i][1]
        return predict_values


def _create_long_network():
    in_layer = keras.Input(shape=(143, ))

    curr_layer = keras.layers.Dense(
        300, kernel_initializer="glorot_uniform",
        activation="sigmoid")(in_layer)

    curr_layer_bn = keras.layers.BatchNormalization()(curr_layer)

    # output layer
    out_layer = keras.layers.Dense(
        36, kernel_initializer="glorot_uniform",
        activation="softmax")(curr_layer_bn)

    # return an instance of the Model class
    return keras.Model(inputs=in_layer, outputs=out_layer)


if __name__ == "__main__":
    long_features = pd.read_csv(
        "./features/long_features.csv", header=None, nrows=160000)
    long_features = long_features.dropna().values

    with open(f"./lime_extended_performance_{sys.argv[1]}.csv", "w") as FILE:
        for j in range(20):
            for i in range(0, 50, 1):
                model = _create_long_network()
                model.load_weights(
                    f"./weights/tmp_long_rla_weights")
                feature_train, feature_test = train_test_split(
                    long_features, test_size=0.2)

                action_train = model.predict(feature_train)
                action_test = model.predict(feature_test)

                pca = PCA(n_components=48)
                feature_train = pca.fit_transform(feature_train)
                feature_test = pca.transform(feature_test)

                lime_model = LIMESimpleModel(cluster_num=i + 1)
                lime_model.fit(
                    X=feature_train,
                    y=action_train,
                    predict_fn=model.predict,
                    labels_num=36)

                lime_action_train = lime_model.predict(feature_train)
                lime_action_test = lime_model.predict(feature_test)

                train_accuracy = np.mean(
                    np.int32(
                        lime_action_train == np.argmax(action_train, axis=1)))
                train_rmse = np.mean(
                    np.int32(
                        lime_action_test == np.argmax(action_test, axis=1)))
                test_accuracy = np.sqrt(
                    np.mean(
                        np.square(lime_action_train -
                                  np.argmax(action_train, axis=1))))
                test_rmse = np.sqrt(
                    np.mean(
                        np.square(lime_action_test -
                                  np.argmax(action_test, axis=1))))

                FILE.write(f"{i + 1}, {train_accuracy}, {test_accuracy}"
                           f", {train_rmse}, {test_rmse}\n")
