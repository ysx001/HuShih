
import argparse
import zmq
import pickle
import numpy as np

#Here's an example dict
ll_msg = np.array(['123', '456', 'lol'])

parser = argparse.ArgumentParser(description='zeromq server/client')
parser.add_argument('--bar')
args = parser.parse_args()

if args.bar:
    # client
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://127.0.0.1:5555')
    socket.send(pickle.dumps(ll_msg))
    print("sent")
    msg = socket.recv_string()
    print(msg)
else:
    # server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://127.0.0.1:5555')
    while True:
        msg = socket.recv()
        deserialized_msg = pickle.loads(msg)
        print("deserialized_msg:", deserialized_msg, type(deserialized_msg))
        if (deserialized_msg == ll_msg).all():
            socket.send_string('ah ha!')
        else:
            socket.send_string("...nah")
