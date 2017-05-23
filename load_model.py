import load
import pickle
import theano
import theano.tensor as T
import math
import chess, chess.pgn
import heapq
import time
import re
import string
import numpy
#import sunfish
import pickle
import random
import traceback

def get_model_from_pickle(fn):
    f = open(fn, "rb")
    Ws, bs = pickle.load(f)
    
    Ws_s, bs_s = load.get_parameters(Ws=Ws, bs=bs)
    x, p = load.get_model(Ws_s, bs_s)
    
    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict

strip_whitespace = re.compile(r"\s+")
translate_pieces = string.maketrans(".pnbrqkPNBRQK", "\x00" + "\x01\x02\x03\x04\x05\x06" + "\x08\x09\x0a\x0b\x0c\x0d")

def sf2array(pos, flip):
    # Create a numpy array from a sunfish representation
    pos = strip_whitespace.sub('', pos) # should be 64 characters now
    pos = pos.translate(translate_pieces)
    m = numpy.fromstring(pos, dtype=numpy.int8)
    if flip:
        m = numpy.fliplr(m.reshape(8, 8)).reshape(64)
    return m


func = get_model_from_pickle('model_aws.pickle')