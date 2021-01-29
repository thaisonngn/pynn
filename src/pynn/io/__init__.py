import gzip
import os
import sys, re
import glob
import random
import threading

import numpy

# Prepare readers for compressed files
readers = {}
try:
    import gzip
    readers['.gz'] = gzip.GzipFile
except ImportError:
    pass
try:
    import bz2
    readers['.bz2'] = bz2.BZ2File
except ImportError:
    pass

def smart_open(filename, mode = 'rb', *args, **kwargs):
    '''
    Opens a file "smartly":
      * If the filename has a ".gz" or ".bz2" extension, compression is handled
        automatically;
      * If the file is to be read and does not exist, corresponding files with
        a ".gz" or ".bz2" extension will be attempted.
    (The Python packages "gzip" and "bz2" must be installed to deal with the
        corresponding extensions)
    '''
    if 'r' in mode and not os.path.exists(filename):
        for ext in readers:
            if os.path.exists(filename + ext):
                filename += ext
                break
    extension = os.path.splitext(filename)[1]
    return readers.get(extension, open)(filename, mode, *args, **kwargs)

def make_context(feature, left, right):
    '''
    Takes a 2-D numpy feature array, and pads each frame with a specified
        number of frames on either side.
    '''
    feature = [feature]
    for i in range(left):
        feature.append(numpy.vstack((feature[-1][0], feature[-1][:-1])))
    feature.reverse()
    for i in range(right):
        feature.append(numpy.vstack((feature[-1][1:], feature[-1][-1])))
    return numpy.hstack(feature)

def shuffle_data(feature, label=None, target=None):
    '''
    Randomly shuffles features and labels in the *same* order.
    '''
    seed = 18877
    numpy.random.seed(seed)
    numpy.random.shuffle(feature)
    if label is not None:
        numpy.random.seed(seed)
        numpy.random.shuffle(label)
        
    if target is not None:
        numpy.random.seed(seed)
        numpy.random.shuffle(target)
