from abc import ABC, abstractmethod
from queue import Queue

class Channel(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def send(self, data):
        pass
    
    @abstractmethod
    def recv(self):
        pass


class IO(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def sender(self):
        pass
    
    @abstractmethod
    def receiver(self):
        pass


class QueueIO(IO):
    
    sender_queue = Queue(maxsize=10)
    receiver_queue = Queue(maxsize=10)
    
    def __init__(self):
        pass

    def sender(self):
        return QueueChannel(QueueIO.sender_queue, QueueIO.receiver_queue)
    
    def receiver(self):
        return QueueChannel(QueueIO.receiver_queue, QueueIO.sender_queue)


class QueueChannel(Channel):
    
    def __init__(self, queue, peer_queue):
        self._queue = queue
        self._peer_queue = peer_queue

    def send(self, data):
        self._peer_queue.put(data)

    def recv(self):
        return self._queue.get()
