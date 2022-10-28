import sys
import queue
import time
from threading import Event, Thread
from concurrent import futures
import grpc

from network.pseudo_duplex_channel_pb2 import Message, Bytes, Empty
from network.pseudo_duplex_channel_pb2_grpc import PseudoDuplexChannelServicer, \
    add_PseudoDuplexChannelServicer_to_server
from network.io import Channel

MAX_MESSAGE_LENGTH = 256 * 1024 * 1024


class GrpcChannelServicer(PseudoDuplexChannelServicer):
    """A pseudo-duplex network server based on GRPC.
    """

    send_queue = queue.Queue(maxsize=20)
    recv_queue = queue.Queue(maxsize=20)

    def __init__(self):
        pass

    def recv(self, request, context):
        """The client request to receive data.
        Arguments:
            self: The instance.
            request: Request data.
            context: Context.
        Returns:
            The message if there's anything to be sent in sending queue, or empty if timeout.
        """
        while True:
            # Poll send_queue for new data
            data = GrpcChannelServicer.send_queue.get(timeout=30)
            if data:
                # Respond with data in send_queue
                msg = Message()
                msg.bytes.data = data
                return msg
            else:
                # Or poll send_queue again if there's nothing and it's timed out
                # No data yet in send_queue
                msg = Message()
                msg.empty = Empty()
                return msg

    def send(self, request, context):
        # Save request data into recv_queue
        GrpcChannelServicer.recv_queue.put(request.bytes.data)
        return Empty()


class GrpcChannelServer(Channel):

    def __init__(self, addr_and_port="[::]:50052"):
        self._servicer_thread = Thread(target=self.serve, args=(addr_and_port,))
        self._servicer_thread.start()
        self._stop_event = Event()

    def serve(self, addr_and_port):
        """Start server to serve.
        Arguments:
            addr_and_port (str): The address and port, default: "[::]:50052"
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)],)
        servicer = GrpcChannelServicer()
        add_PseudoDuplexChannelServicer_to_server(servicer, server)
        server.add_insecure_port(addr_and_port)
        server.start()
        print("Server started at %s" % addr_and_port)
        self._stop_event.wait()
        if not GrpcChannelServicer.send_queue.empty():
            time.sleep(3)
        server.stop(grace=10)
        # server.wait_for_termination()

    def stop(self):
        self._stop_event.set()

    def recv(self):
        # Pick data from recv_queue, or wait for new data to be ready
        data = GrpcChannelServicer.recv_queue.get()
        return data

    def send(self, data):
        # Put data to send_queue, and get ready for client recv request
        GrpcChannelServicer.send_queue.put(data)


if __name__ == "__main__":
    server = GrpcChannelServer()
    server.send(b"hello")
    print("Sent %s to client" % b"hello")
    data = server.recv()
    print("Received %s from client" % data)

    server.stop()
