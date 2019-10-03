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

from .rl import *
import sys
from .dqn import *
from .dt import *
from .log import *
from .pensieve_viper_env import *
from ..pensieve_test.load_trace import *

TRAIN_TRACES = './pensieve/pensieve_test/cooked_new_traces/'

def learn_dt(leaf_nodes):
    # parameters
    log_fname = './pensieve/pensieve_viper/pensieve_dt.log'
    model_path = './pensieve/pensieve_test/models/pretrain_linear_reward.ckpt'
    n_batch_rollouts = 10
    max_samples = 200000
    max_iters = 100
    train_frac = 0.8
    is_reweight = False
    n_test_rollouts = 50
    save_dirname = './pensieve/pensieve_viper/decision_tree'
    save_fname = 'pensieve_dt_policy_' + leaf_nodes + '.pk'
    save_viz_fname = 'pensieve_dt_policy_' + leaf_nodes + '.svg'
    is_train = True

    parameters = {}
    parameters['MILLISECONDS_IN_SECOND'] = 1000.0
    parameters['B_IN_MB'] = 1000000.0
    parameters['BITS_IN_BYTE'] = 8.0
    parameters['RANDOM_SEED'] = 42
    parameters['VIDEO_CHUNCK_LEN'] = 4000.0  # millisec, every time add this amount to buffer
    parameters['BITRATE_LEVELS'] = 6
    parameters['TOTAL_VIDEO_CHUNCK'] = 48
    parameters['BUFFER_THRESH'] = 60.0 * parameters['MILLISECONDS_IN_SECOND']  # millisec, max buffer limit
    parameters['DRAIN_BUFFER_SLEEP_TIME'] = 500.0  # millisec
    parameters['PACKET_PAYLOAD_PORTION'] = 0.95
    parameters['LINK_RTT'] = 80  # millisec
    parameters['PACKET_SIZE'] = 1500  # bytes
    parameters['NOISE_LOW'] = 0.9
    parameters['NOISE_HIGH'] = 1.1
    parameters['VIDEO_SIZE_FILE'] = './pensieve/pensieve_test/video_size_'
    parameters['S_INFO'] = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
    parameters['S_LEN'] = 8  # take how many frames in the past
    parameters['A_DIM'] = 6
    parameters['ACTOR_LR_RATE'] = 0.0001
    parameters['CRITIC_LR_RATE'] = 0.001
    parameters['VIDEO_BIT_RATE'] = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
    parameters['BUFFER_NORM_FACTOR'] = 10.0
    parameters['CHUNK_TIL_VIDEO_END_CAP'] = 48.0
    parameters['M_IN_K'] = 1000.0
    parameters['REBUF_PENALTY'] = 4.3  # 1 sec rebuffering -> 3 Mbps
    parameters['SMOOTH_PENALTY'] = 1
    parameters['DEFAULT_QUALITY'] = 1  # default video quality without agent
    parameters['RANDOM_SEED'] = 42
    parameters['RAND_RANGE'] = 1000
    parameters['SUMMARY_DIR'] = './results'
    parameters['LOG_FILE'] = './results/log_sim_rl'

    feature_names = ['last_qoe', 'last_buffer']
    for i in range(parameters['S_LEN']):
        feature_names.append('throughput_' + str(i))
    for i in range(parameters['S_LEN']):
        feature_names.append('download_' + str(i))
    for i in range(parameters['A_DIM']):
        feature_names.append('next_size_' + str(i))
    feature_names.append('remain')


    all_cooked_time, all_cooked_bw, all_file_names = load_trace(TRAIN_TRACES)
    parameters['ALL_FILE_NAMES'] = all_file_names

    # Logging
    set_file(log_fname)

    # Data structures
    env = Environment(all_cooked_time=all_cooked_time,
                      all_cooked_bw=all_cooked_bw,
                      parameters=parameters)
    teacher = DQNPolicy(env, model_path, parameters, n_batch_rollouts)

    max_leaf_nodes = None
    if leaf_nodes != 'None':
        max_leaf_nodes = int(leaf_nodes)
    max_depth = None
    student = DTPolicy(max_depth, max_leaf_nodes, parameters)

    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, max_iters, n_batch_rollouts, max_samples,
                               train_frac, is_reweight, n_test_rollouts, parameters, feature_names)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname, feature_names=feature_names)
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    rew = test_policy(env, student, n_test_rollouts, parameters)
    log('Final reward: {}'.format(rew), INFO)
    log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)


def to_state():
        # TODO: extract states from the pensieve environment
        pass


def bin_acts():
    # parameters
    seq_len = 10
    n_rollouts = 10
    log_fname = 'pong_options.log'
    model_path = 'model-atari-pong-1/saved'

    # Logging
    set_file(log_fname)

    # Data structures
    env = None  # TODO: update env
    teacher = DQNPolicy(env, model_path)

    # Action sequences
    seqs = get_action_sequences(env, teacher, seq_len, n_rollouts)

    for seq, count in seqs:
        log('{}: {}'.format(seq, count), INFO)


def print_size():
    # parameters
    dirname = 'results/run9'
    fname = 'dt_policy.pk'

    # Load decision tree
    dt = load_dt_policy(dirname, fname)

    # Size
    print(dt.tree.tree_.node_count)


if __name__ == '__main__':
    learn_dt(sys.argv[1])
