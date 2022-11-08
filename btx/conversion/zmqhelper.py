import zmq
import numpy as np
import zlib, pickle

"""
This script was taken directly from Mona's xtc1to2 repository and 
copied unchanged into btx for convenience for the exafel demo.
"""

class ZmqSender:
    """A helper for sending messages using pyzmq.
    This wraps codes that came from example how pyzmq serializes/
    deserializes python objects. For sending multiple object,
    use send/recv_zipped_pickle method pair.
    """

    zmq_socket = None

    def __init__(self, socket):
        """Bind to socket (e.g. tcp://127.0.0.1:5557) with PUSH"""
        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.PUSH)
        self.zmq_socket.bind(socket)

    def send_zipped_pickle(self, obj, flags=0, protocol=-1):
        """pickle an object, and zip the pickle before sending it"""
        p = pickle.dumps(obj, protocol)
        z = zlib.compress(p)
        return self.zmq_socket.send(z, flags=flags)

    def send_array(self, A, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.zmq_socket.send_json(md, flags | zmq.SNDMORE)
        return self.zmq_socket.send(A, flags, copy=copy, track=track)


class ZmqReceiver:
    """A helper for receiving messages using pyzmq."""

    zmq_socket = None

    def __init__(self, socket):
        """Bind to socket (e.g. tcp://127.0.0.1:5557) with PULL"""
        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.PULL)
        self.zmq_socket.connect(socket)

    def recv_zipped_pickle(self, flags=0, protocol=-1):
        """inverse of send_zipped_pickle"""
        z = self.zmq_socket.recv(flags)
        p = zlib.decompress(z)
        return pickle.loads(p)

    def recv_array(self, md, flags=0, copy=True, track=False):
        """recv a numpy array"""
        msg = self.zmq_socket.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)
        A = np.frombuffer(buf, dtype=md["dtype"])
        return A.reshape(md["shape"])
