import queue
import grpc

from network.pseudo_duplex_channel_pb2 import Bytes, Empty, Message
from network.pseudo_duplex_channel_pb2_grpc import PseudoDuplexChannelStub
from network.io import Channel

MAX_MESSAGE_LENGTH = 256 * 1024 * 1024


class GrpcChannelClient(Channel):
    """A pseudo-duplex network client based on GRPC.
    """

    def __init__(self, addr_and_port="localhost:50052"):
        self._channel = None
        self._stub = None
        self._queue = queue.Queue(maxsize=20)
        self.connect(addr_and_port)

    def connect(self, addr_and_port):

        self._channel = grpc.insecure_channel(addr_and_port, options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        self._stub = PseudoDuplexChannelStub(self._channel)
        print("Created stub to %s" % addr_and_port)

    def disconnect(self):
        self._channel.close()

    def recv(self):
        while True:
            # Initiate a recv request
            resp = self._stub.recv(Empty())
            if hasattr(resp, "bytes"):
                # The recv request has been fullfiled. Return the received data.
                # Otherwise, will initiate a new recv request again.
                return resp.bytes.data

    def send(self, data):
        msg = Message()
        msg.bytes.data = data
        # Send the data as normal send request
        self._stub.send(msg)


if __name__ == "__main__":
    client = GrpcChannelClient(addr_and_port="localhost:50052")
    client.send(b"world")
    print("Sent %s to server" % b"world")
    data = client.recv()
    print("Received %s from server" % data)

    client.disconnect()
