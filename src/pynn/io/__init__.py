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

def down_sample(feature, label=None, rand=True):
    l = feature.shape[0] // 3 *3
    j = random.randint(0,1) if rand else 0
    feature = [feature[j:l:3,:], feature[j+1:l:3,:]]
    feature = numpy.hstack(feature)

    if label is not None:
        label = label[j+1:l:3][:feature.shape[0]]
        return (feature, label)
    else:
        return feature

def down_sample_2x(feature):
    idx = list(range(0, feature.shape[0], 3))
    feature = numpy.delete(feature, idx, axis=0)
    
    feature = feature[:(feature.shape[0]//4)*4,:]
    feature = feature.reshape(feature.shape[0]//4, feature.shape[1]*4)

    return feature

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
                
class DataPool(object):

    class ReadingThread(threading.Thread):
        def __init__(self, reader, item_size, pool_size):
            threading.Thread.__init__(self)
            
            self.reader = reader
            self.item_size = item_size
            self.pool_size = pool_size

            self.buffer = []
            self.end_reading = False
     
        def run(self):
            for i in range(self.pool_size):
                if not self.reader.available():
                    self.end_reading = True
                    break
                item = self.reader.next(self.item_size)
                self.buffer.append(item)

    def __init__(self, reader, item_size, pool_size=4000):
        self.reader = reader
        self.item_size = item_size
        self.pool_size = pool_size
        
    def initialize(self):
        self.reader.initialize()
     
        self.items = []     
        self.index = 0
        self.thread = None
        self.end_reading = False

    def available(self):
        if not self.end_reading and self.thread is None:
            self.thread = DataPool.ReadingThread(self.reader, self.item_size, self.pool_size)
            self.thread.start()

        if self.index >= len(self.items) and self.thread is not None:
            self.thread.join()
            self.items = self.thread.buffer
            self.end_reading = self.thread.end_reading
            self.index = 0
            self.thread = None

        return self.index < len(self.items)
    
    def next(self):
        item = self.items[self.index]
        self.index += 1
        return item
