import sys
from multiprocessing import Manager, Pool, Process

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

import keras
from tqdm import tqdm

EPSILON = 1e-7


class MixtureLinearRegressionModel:
    def __init__(self, component, verbose=False, max_iteration=500):
        self.verbose = verbose
        self.max_iteration = max_iteration
        self.component = component
        self.linear_models = []
        self.pi_ = None
        self.sigma_ = None
        self.model = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.numSample = X.shape[0]
        self.numFeature = X.shape[1]

        self.__initInteration()
        for i in range(self.max_iteration):
            if not self.__E_step():
                self.__M_step()
            else:
                break

    def __E_step(self):
        linear_model = np.zeros((self.numSample, self.component))
        for i in range(self.component):
            linear_model[:, i] = self.model[i].predict(self.X)

        linear_offset = self.y.reshape(-1, 1) - linear_model
        probability = -0.5 * np.log(self.sigma_ + EPSILON) - (
            linear_offset**2) / (2 * self.sigma_ + EPSILON) + np.log(self.pi_ +
                                                                     EPSILON)

        updated_cluster_result = np.argmax(probability, axis=1)
        self.local_prediction = linear_model[np.arange(
            0, linear_model.shape[0]), updated_cluster_result]
        diff_cluster = np.sum(
            np.abs(updated_cluster_result - self.cluster_result))
        if diff_cluster == 0:
            return True
        else:
            self.cluster_result = updated_cluster_result
            return False

    def __M_step(self):
        for i in range(self.component):
            linear_reg = LinearRegression()
            inds = np.where(self.cluster_result == i)
            if len(inds[0]) != 0:

                linear_reg.fit(self.X[inds], self.y[inds])
                self.pi_[:, i] = np.sum(
                    np.int32(self.cluster_result == i)) / self.numSample
                self.model[i] = linear_reg
                self.sigma_[:, i] = np.mean(
                    (self.y[inds] - linear_reg.predict(self.X[inds]))**2)

            else:
                self.pi_[:, i] = 0.0

        assert (np.sum(self.pi_) - 1 < 1e-3)

    def __initInteration(self):
        assert self.X is not None

        self.pi_ = np.zeros((1, self.component))
        self.sigma_ = np.zeros((1, self.component))
        self.coef_ = np.zeros((self.numFeature, self.component))
        self.intercept_ = np.zeros_like(self.sigma_)
        self.cluster_result = np.zeros(self.numSample)

        randDist = np.arange(0, self.X.shape[0])
        np.random.shuffle(randDist)
        for i in range(self.component):
            linear_reg = LinearRegression()
            pointRange = randDist[int(i * self.numSample / self.component):int(
                (i + 1) * self.numSample / self.component)]
            self.cluster_result[pointRange] = i
            linear_reg.fit(self.X[pointRange], self.y[pointRange])
            self.pi_[:, i] = len(pointRange) / self.numSample
            self.sigma_[:, i] = np.mean(
                (self.y[pointRange] - linear_reg.predict(self.X[pointRange]))
                **2)
            self.model.append(linear_reg)

    def predict(self, row):
        linear_model = np.zeros((row.shape[0], self.component))
        for i in range(self.component):
            if self.pi_[0, i] - 0 < 1e-6:
                linear_model[:, i] = 0
            else:
                linear_model[:, i] = self.model[i].predict(row)

        return np.sum(self.pi_ * linear_model, axis=1)

    def local_prediction(self):
        return self.local_prediction


class LEMNASimpleModel:
    def __init__(self, cluster_num, cluster_method=KMeans, random_state=None):
        self.random_state = check_random_state(random_state)
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method(
            n_clusters=cluster_num, random_state=self.random_state)
        self.models = []

    def fit(self, X, y, lemna_component, predict_fn, labels_num):
        self.labels_num = labels_num
        self.cluster_labels = self.cluster_method.fit_predict(X)
        self.num_features = X.shape[1]
        self.local_prediction = np.zeros((X.shape[0], labels_num))

        for i in range(self.cluster_num):
            simplified_models = []

            inds = np.where(self.cluster_labels == i)
            # coef_ is a 3-d matrix feature_num * lemna_component * labels_num
            # intercept is a 2-d matrix lemna_component * labels_num

            for idx in range(labels_num):
                simplified_model = MixtureLinearRegressionModel(
                    component=lemna_component, verbose=(idx == 0))
                simplified_model.fit(X[inds], np.squeeze(y[inds, idx]))

                simplified_models.append(simplified_model)

                self.local_prediction[inds,
                                      idx] = simplified_model.local_prediction

            self.models.append(simplified_models)

    def predict(self, x):
        cluster_result = self.cluster_method.predict(x)
        prediction_result = np.zeros(x.shape[0])

        for i in range(self.cluster_num):
            inds = np.where(cluster_result == i)
            if not len(inds[0]):
                continue

            predict_values = np.zeros((len(inds[0]), self.labels_num))
            for idx in range(self.labels_num):
                predict_values[:, idx] = self.models[i][idx].predict(x[inds])
            prediction_result[inds] = np.argmax(predict_values, axis=1)

        return prediction_result


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


def calculate_precess(taskQueue, resultQueue):
    long_features = pd.read_csv(
        "./features/long_features.csv",
        sep=',',
        header=None,
        encoding='utf-8')
    model = _create_long_network()
    model.load_weights(f"./weights/tmp_long_rla_weights")
    while True:
        cluster_number = taskQueue.get(None)
        if cluster_number == "":
            break
        feature_train, feature_test = train_test_split(
            long_features, test_size=0.2)

        action_train = model.predict(feature_train)
        action_test = model.predict(feature_test)

        print("After NN prediction...")

        pca = PCA(n_components=48)
        feature_train = pca.fit_transform(feature_train)
        feature_test = pca.transform(feature_test)

        print("After PCA...")

        lemna_model = LEMNASimpleModel(cluster_num=cluster_number)
        lemna_model.fit(
            X=feature_train,
            y=action_train,
            lemna_component=2,
            predict_fn=model.predict,
            labels_num=36)

        lemna_action_train = np.argmax(lemna_model.local_prediction, axis=1)
        lemna_action_test = lemna_model.predict(feature_test)

        print("After LEMNA prediction...")

        train_accuracy = np.mean(
            np.int32(lemna_action_train == np.argmax(action_train, axis=1)))
        test_accuracy = np.mean(
            np.int32(lemna_action_test == np.argmax(action_test, axis=1)))
        train_rmse = np.sqrt(
            np.mean(
                np.square(lemna_action_train -
                          np.argmax(action_train, axis=1))))
        test_rmse = np.sqrt(
            np.mean(
                np.square(lemna_action_test - np.argmax(action_test, axis=1))))

        resultQueue.put((cluster_number, train_accuracy, test_accuracy,
                         train_rmse, test_rmse))


def update_process(resultQueue, totalCount):
    progress = tqdm(total=totalCount, desc="Total", unit="B", unit_scale=True)
    with open("./lemna_extended_performance.csv", "w") as FILE:
        while True:
            res = resultQueue.get(1)
            if res != "":
                FILE.write(",".join([str(item) for item in res]) + "\n")
                FILE.flush()
                progress.n += 1
                progress.update(0)
            else:
                break


if __name__ == "__main__":
    manager = Manager()
    taskQueue = manager.Queue()
    resultQueue = manager.Queue()

    workers = int(sys.argv[1])

    pool = Pool(processes=workers)

    for _ in range(workers):
        pool.apply_async(calculate_precess, args=(taskQueue, resultQueue))

    num = 10
    cluster_list = list(range(1, 41, 1)) * num
    for index in cluster_list:
        taskQueue.put(index)

    for _ in range(workers):
        taskQueue.put("")

    update_Process = Process(
        target=update_process, args=(resultQueue, len(cluster_list)))

    update_Process.start()
    pool.close()
    pool.join()
    resultQueue.put("")
    update_Process.join()
