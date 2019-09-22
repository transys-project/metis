# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle as pk
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from .log import *
from .ccp import *
import pydotplus


def accuracy(policy, obss, acts):
    return np.mean(acts == policy.predict(obss))


def split_train_test(obss, acts, copies, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    copies_train = copies[idx[:n_train]]
    copies_test = copies[idx[n_train:]]
    return obss_train, acts_train, copies_train, obss_test, acts_test, copies_test


def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pk.dump(dt_policy, f)
    f.close()


def save_dt_policy_viz(dt_policy, dirname, fname, feature_names):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    dot_data = StringIO()
    export_graphviz(dt_policy.tree, out_file=dot_data, feature_names=feature_names, filled=True)
    out_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    out_graph.write_svg(dirname + '/' + fname)


def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pk.load(f)
    f.close()
    return dt_policy


class DTPolicy:
    def __init__(self, max_depth, max_leaf_nodes, parameters):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.tree = None
        self.parameters = parameters
    
    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=None, max_leaf_nodes=None)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac, copies, feature_names, env):
        obss_train, acts_train, copies_train, obss_test, acts_test, copies_test = split_train_test(obss, acts, copies, train_frac)
        self.fit(obss_train, acts_train)

        # prune
        print('-' * 50)
        print('Prune begins...')
        classes = []
        for x in acts_train:
            item = str(x)
            if item not in classes : classes.append(item)
        classes.sort()
        prune_tree = prune(tree=self.tree, features=feature_names, classes=classes, x_test=obss_test, y_test=acts_test, \
                           max_leaf_nodes=self.max_leaf_nodes, x_train=obss_train, copies_train=copies_train, \
                           copies_test=copies_test, parameters=self.parameters, env=env)
        acts_prune = prune_tree.predict(obss_test)
        accuracy_prune = np.mean(acts_prune == acts_test)
        print('Prune ends...')
        print('Accuracy by pruning tree', accuracy_prune)
        self.tree = prune_tree
        
        log('Train accuracy: {}'.format(accuracy(self, obss_train, acts_train)), INFO)
        log('Test accuracy: {}'.format(accuracy(self, obss_test, acts_test)), INFO)

    def predict(self, obss):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth, self.max_leaf_nodes, self.parameters)
        clone.tree = self.tree
        return clone
