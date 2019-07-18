import numpy as np
import tensorflow as tf

from mpi4py import MPI

# from feature_extractor import Feature_extractor



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lambda', type=float, default=0.95) # basic example on how to use the parser

    #args = parser.parse_args()
    #print(args.__dict__)
    # with tf.Session() as sess:
    #     asf = Feature_extractor()
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank() # debatable wether i want to think about mpi now..., maybe if it works, put in the effort to make it run on mpi