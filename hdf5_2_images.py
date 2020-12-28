import os, sys, time, argparse
import queue
import numpy as np
import h5py

DVS_SHAPE = (260, 346)

f = h5py.File('/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/DDD17/rec1487779465.hdf5', 'r')
print(type(f))
print(f['/dvs']['data'][:,1])
