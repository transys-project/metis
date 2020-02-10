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

from a3c import *
from log import *


def get_reward(bit_rate, rebuf, last_bit_rate, parameters, reward_type):
    if reward_type == 'LIN':
        reward = parameters['VIDEO_BIT_RATE'][bit_rate] / parameters['M_IN_K'] - parameters['REBUF_PENALTY'] * rebuf - \
                 parameters['SMOOTH_PENALTY'] * np.abs(
            parameters['VIDEO_BIT_RATE'][bit_rate] - parameters['VIDEO_BIT_RATE'][last_bit_rate]) / parameters['M_IN_K']
    elif reward_type == 'LOG':
        log_bit_rate = np.log(parameters['VIDEO_BIT_RATE'][bit_rate] / float(parameters['VIDEO_BIT_RATE'][-1]))
        log_last_bit_rate = np.log(
            parameters['VIDEO_BIT_RATE'][last_bit_rate] / float(parameters['VIDEO_BIT_RATE'][-1]))
        reward = log_bit_rate - parameters['REBUF_PENALTY'] * rebuf - parameters['SMOOTH_PENALTY'] * np.abs(
            log_bit_rate - log_last_bit_rate)
    elif reward_type == 'HD':
        reward = parameters['HD_REWARD'][bit_rate] - parameters['REBUF_PENALTY'] * rebuf - parameters[
            'SMOOTH_PENALTY'] * np.abs(parameters['HD_REWARD'][bit_rate] - parameters['HD_REWARD'][last_bit_rate])
    else:
        reward = None
    return reward


def get_rollout(env, policy, parameters, is_student=True):
    rollout = []
    time_stamp = 0
    last_bit_rate = parameters['DEFAULT_QUALITY']
    bit_rate = parameters['DEFAULT_QUALITY']
    action_vec = np.zeros(parameters['A_DIM'])
    action_vec[bit_rate] = 1
    s_batch = [np.zeros((parameters['S_INFO'], parameters['S_LEN']))]
    a_batch = [action_vec]
    r_batch = []
    # entropy_record = []

    video_count = 0
    while True:
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, \
        next_video_chunk_sizes, end_of_video, video_chunk_remain = env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        reward = get_reward(bit_rate, rebuf, last_bit_rate, parameters, 'LIN')
        r_batch.append(reward)

        last_bit_rate = bit_rate

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((parameters['S_INFO'], parameters['S_LEN']))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = parameters['VIDEO_BIT_RATE'][bit_rate] / float(np.max(parameters['VIDEO_BIT_RATE']))  # last QoE
        state[1, -1] = buffer_size / parameters['BUFFER_NORM_FACTOR']  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / parameters['M_IN_K']  # kilo byte / ms
        state[3, -1] = float(delay) / parameters['M_IN_K'] / parameters['BUFFER_NORM_FACTOR']  # 10 sec
        state[4, :parameters['A_DIM']] = np.array(next_video_chunk_sizes) / parameters['M_IN_K'] / parameters['M_IN_K']  # MB
        state[5, -1] = np.minimum(video_chunk_remain, parameters['CHUNK_TIL_VIDEO_END_CAP']) / \
                       float(parameters['CHUNK_TIL_VIDEO_END_CAP'])

        serilized_state = []
        serilized_state.append(state[0, -1])
        serilized_state.append(state[1, -1])
        for i in range(parameters['S_LEN']):
            serilized_state.append(state[2, i])
        for i in range(parameters['S_LEN']):
            serilized_state.append(state[3, i])
        for i in range(parameters['A_DIM']):
            serilized_state.append(state[4, i])
        serilized_state.append(state[5, -1])

        if is_student:
            bit_rate = policy.predict(np.array(serilized_state).reshape(1, -1))[0]
        else:
            action_prob = policy.predict(np.reshape(state, (1, parameters['S_INFO'], parameters['S_LEN'])))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, parameters['RAND_RANGE']) /
                        float(parameters['RAND_RANGE'])).argmax()
        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

        # entropy_record.append(compute_entropy(action_prob[0]))

        if end_of_video:
            last_bit_rate = parameters['DEFAULT_QUALITY']
            bit_rate = parameters['DEFAULT_QUALITY']  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((parameters['S_INFO'], parameters['S_LEN'])))
            a_batch.append(action_vec)
            # entropy_record = []

            # print("video count", video_count)
            # video_count += 1
            #
            # if video_count >= len(parameters['ALL_FILE_NAMES']):
            #     break

            break
        para_copy = [int(env.trace_idx), int(env.mahimahi_ptr), int(env.video_chunk_counter), env.buffer_size]
        rollout.append((state, bit_rate, reward, serilized_state, para_copy))
    return rollout


def get_rollouts(env, policy, n_batch_rollouts, parameters, is_student=True):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout(env, policy, parameters, is_student))
    return rollouts


def _sample(obss, acts, qs, copies, max_pts, is_reweight):
    # Step 1: Compute probabilities
    ps = np.max(qs, axis=1) - np.min(qs, axis=1)
    ps = ps / np.sum(ps)

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(len(obss), size=max_pts)  # min(max_pts, np.sum(ps > 0)), replace=False)

    # Step 3: Obtain sampled indices
    return obss[idx], acts[idx], qs[idx], copies[idx]


class TransformerPolicy:
    def __init__(self, policy, state_transformer):
        self.policy = policy
        self.state_transformer = state_transformer

    def predict(self, obss):
        return self.policy.predict(np.array([self.state_transformer(obs) for obs in obss]))


def test_policy(env, policy, n_test_rollouts, parameters):
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        student_trace = get_rollout(env, policy, parameters)
        cum_rew += sum((rew for _, _, rew, _, _ in student_trace))
    return cum_rew / n_test_rollouts


def identify_best_policy(env, policies, n_test_rollouts, parameters):
    log('Initial policy count: {}'.format(len(policies)), INFO)
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1)/2)
        log('Current policy count: {}'.format(n_policies), INFO)

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            new_rew = test_policy(env, policy, n_test_rollouts, parameters)
            new_policies.append((policy, new_rew))
            log('Reward update: {} -> {}'.format(rew, new_rew), INFO)

        policies = new_policies

    if len(policies) != 1:
        raise Exception()

    return policies[0][0]


def _get_action_sequences_helper(trace, seq_len):
    acts = [act for _, act, _ in trace]
    seqs = []
    for i in range(len(acts) - seq_len + 1):
        seqs.append(acts[i:i+seq_len])
    return seqs


def get_action_sequences(env, policy, seq_len, n_rollouts, parameters):
    # Step 1: Get action sequences
    seqs = []
    for _ in range(n_rollouts):
        trace = get_rollout(env, policy, parameters)
        seqs.extend(_get_action_sequences_helper(trace, seq_len))

    # Step 2: Bin action sequences
    counter = {}
    for seq in seqs:
        s = str(seq)
        if s in counter:
            counter[s] += 1
        else:
            counter[s] = 1

    # Step 3: Sort action sequences
    seqs_sorted = sorted(list(counter.items()), key=lambda pair: -pair[1])

    return seqs_sorted


def train_dagger(env, teacher, student, max_iters, n_batch_rollouts, max_samples, train_frac,
                 is_reweight, n_test_rollouts, parameters, feature_names):
    # Step 0: Setup
    np.random.seed(parameters['RANDOM_SEED'])
    obss, acts, qs, serialized_obss = [], [], [], []
    students = []
    
    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env=env, policy=teacher, n_batch_rollouts=n_batch_rollouts,
                         parameters=parameters, is_student=False)

    obss.extend((obs for obs, _, _, _, _ in trace))
    acts.extend((act for _, act, _, _, _ in trace))
    serialized_obss.extend((serialized_obs for _, _, _, serialized_obs, _ in trace))
    qs.extend(teacher.predict_q(np.array([obs for obs, _, _, _, _ in trace])))

    copies = []
    copies.extend((copy for _, _, _, _, copy in trace))

    # Step 2: Dagger outer loop
    for i in range(max_iters):
        log('Iteration {}/{}'.format(i, max_iters), INFO)

        # Step 2a: Train from a random subset of aggregated data
        cur_serialized_obss, cur_acts, cur_qs, cur_copies = _sample(np.array(serialized_obss), np.array(acts), np.array(qs), np.array(copies),
                                                        max_samples, is_reweight)

        # cur_serialized_obss = np.array(serialized_obss)
        # cur_acts = np.array(acts)
        # cur_qs = np.array(qs)
        # cur_copies = np.array(copies)
        log('Training student with {} points'.format(len(cur_serialized_obss)), INFO)
        student.train(cur_serialized_obss, cur_acts, train_frac, cur_copies, feature_names, env)

        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env=env, policy=student, n_batch_rollouts=n_batch_rollouts, parameters=parameters)
        student_obss = [obs for obs, _, _, _, _ in student_trace]
        student_serialized_obss = [serialized_obss for _, _, _, serialized_obss, _ in student_trace]
        student_copies = [copy for _, _, _, _, copy in student_trace]
        
        # Step 2c: Query the oracle for supervision
        teacher_qs = teacher.predict_q(student_obss)
        # at the interface level, order matters, since teacher.predict may run updates
        # teacher_acts = teacher.predict(student_obss)
        teacher_acts = []
        for obs in student_obss:
            para = np.reshape(obs, (1, parameters['S_INFO'], parameters['S_LEN']))
            action_prob = teacher.predict(para)
            action_cumsum = np.cumsum(action_prob)
            teacher_acts.append((action_cumsum > np.random.randint(1, parameters['RAND_RANGE']) /
                                 float(parameters['RAND_RANGE'])).argmax())

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend(student_obss)
        acts.extend(teacher_acts)
        qs.extend(teacher_qs)
        serialized_obss.extend(student_serialized_obss)
        copies.extend(student_copies)

        # Step 2e: Estimate the reward
        cur_rew = sum((rew for _, _, rew, _, _ in student_trace)) / n_batch_rollouts
        log('Student reward: {}'.format(cur_rew), INFO)

        students.append((student.clone(), cur_rew))

    max_student = identify_best_policy(env, students, n_test_rollouts, parameters)

    return max_student
