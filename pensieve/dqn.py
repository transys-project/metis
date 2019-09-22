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
from ..pensieve_test.a3c import *
from .pensieve_viper_env import *
import copy
import math
from .rl import *
import tensorflow as tf
from .rl import *
import tensorflow.contrib.layers as layers


class DQNPolicy:
    def __init__(self, env, model_path, parameters, n_batch_rollouts):
        # Setup
        self.env = env
        self.model_path = model_path
        self.sess = tf.Session()
        self.parameters = parameters
        self.n_batch_rollouts = n_batch_rollouts

        self.actor = ActorNetwork(self.sess, state_dim=[parameters['S_INFO'], parameters['S_LEN']],
                                  action_dim=parameters['A_DIM'], learning_rate=parameters['ACTOR_LR_RATE'])
        self.critic = CriticNetwork(self.sess, state_dim=[parameters['S_INFO'], parameters['S_LEN']],
                                    learning_rate=parameters['CRITIC_LR_RATE'])
        self.sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(self.sess, self.model_path)  # load model

    def predict(self, states):
        return self.actor.predict(states)

    def predict_q(self, states):
        # todo: compute q values from critic networks

        # states -> [470, 6], q_values [470*6, 1]
        # 1. Get a copy of Environment
        env_copy = copy.deepcopy(self.env)

        # 2. Get action according to states, states + action -> next_states
        trace = get_rollouts(env=env_copy, policy=self, n_batch_rollouts=self.n_batch_rollouts,
                             parameters=self.parameters, is_student=False)
        rewards = [reward for _, _, reward, _, _ in trace]
        rewards = np.array(rewards).reshape((len(rewards), 1))

        next_v_states = [obs for obs, _, _, _, _ in trace]
        next_v_values = self.critic.predict(next_v_states)

        q_values = np.tile(next_v_values, (len(self.parameters['VIDEO_BIT_RATE']), 1)) + np.tile(rewards, (len(self.parameters['VIDEO_BIT_RATE']), 1))

        v_values = self.critic.predict(states)
        q_values = np.tile(v_values, (len(self.parameters['VIDEO_BIT_RATE']), 1))

        # q_values = []
        # for state in states:
        #     action_prob = self.actor.predict(np.reshape(state, (1, self.parameters['S_INFO'], self.parameters['S_LEN'])))
        #     for prob in action_prob[0] : q_values.append([np.log(prob)])
        # assert(len(q_values) == 470 * 6)

        return q_values
