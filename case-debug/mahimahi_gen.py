import argparse
import numpy as np

BASE_BW = 12000

def gen(args):
    timestamp = 0
    interval = BASE_BW / args.bw
    residual = 0
    with open('./mahimahi_traces/' + str(args.bw) + 'kbps', 'w') as f:
        while timestamp < max(200, 2*interval+1):
            timestamp += int(np.floor(residual))
            residual -= np.floor(residual)
            f.write(str(timestamp) + '\n')
            residual += interval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bw', type=int)
    args = parser.parse_args()
    gen(args)