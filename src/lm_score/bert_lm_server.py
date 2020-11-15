import argparse
import zmq
import pickle
import numpy as np
from bert_lm import get_sentences_scores

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://127.0.0.1:5555')
    while True:
        msg = socket.recv()
        deserialized_msg = pickle.loads(msg)
        print("deserialized_msg:", deserialized_msg, type(deserialized_msg))
        results = get_sentences_scores(deserialized_msg["curr_iter_decoded"], deserialized_msg["ppl_values"])
        socket.send(pickle.dumps(results))
