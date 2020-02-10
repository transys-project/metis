import numpy as np


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, parameters):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(parameters['RANDOM_SEED'])

        self.parameters = parameters
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(parameters['BITRATE_LEVELS']):
            self.video_size[bitrate] = []
            with open(parameters['VIDEO_SIZE_FILE'] + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < self.parameters['BITRATE_LEVELS']

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]
        
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes
        
        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] * self.parameters['B_IN_MB'] / \
                         self.parameters['BITS_IN_BYTE']
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * self.parameters['PACKET_PAYLOAD_PORTION']

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / self.parameters['PACKET_PAYLOAD_PORTION']
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= self.parameters['MILLISECONDS_IN_SECOND']
        delay += self.parameters['LINK_RTT']

        # add a multiplicative noise to the delay
        delay *= np.random.uniform(self.parameters['NOISE_LOW'], self.parameters['NOISE_HIGH'])

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += self.parameters['VIDEO_CHUNCK_LEN']

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > self.parameters['BUFFER_THRESH']:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.parameters['BUFFER_THRESH']
            sleep_time = np.ceil(drain_buffer_time / self.parameters['DRAIN_BUFFER_SLEEP_TIME']) * \
                         self.parameters['DRAIN_BUFFER_SLEEP_TIME']
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / self.parameters['MILLISECONDS_IN_SECOND']:
                    self.last_mahimahi_time += sleep_time / self.parameters['MILLISECONDS_IN_SECOND']
                    break
                sleep_time -= duration * self.parameters['MILLISECONDS_IN_SECOND']
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.parameters['TOTAL_VIDEO_CHUNCK'] - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.parameters['TOTAL_VIDEO_CHUNCK']:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(self.parameters['BITRATE_LEVELS']):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / self.parameters['MILLISECONDS_IN_SECOND'], \
            rebuf / self.parameters['MILLISECONDS_IN_SECOND'], \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain
