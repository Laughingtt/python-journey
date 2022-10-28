# 先定一个node的类
class Node():  # value + next
    def __init__(self, value=None, next=None):
        self._value = value
        self._next = next

    def getValue(self):
        return self._value

    def getNext(self):
        return self._next

    def setValue(self, new_value):
        self._value = new_value

    def setNext(self, new_next):
        self._next = new_next


# 实现Linked List及其各类操作方法
class LinkedList():
    def __init__(self):  # 初始化链表为空表
        self._head = Node()
        self._tail = None
        self._length = 0

    # 检测是否为空
    def isEmpty(self):
        return self._head == None

        # add在链表前端添加元素:O(1)

    def add(self, value):
        newnode = Node(value, None)  # create一个node（为了插进一个链表）
        newnode.setNext(self._head)
        self._head = newnode

    # append在链表尾部添加元素:O(n)
    def append(self, value):
        newnode = Node(value)
        if self.isEmpty():
            self._head = newnode  # 若为空表，将添加的元素设为第一个元素
        else:
            current = self._head
            while current.getNext() != None:
                current = current.getNext()  # 遍历链表
            current.setNext(newnode)  # 此时current为链表最后的元素

    # search检索元素是否在链表中
    def search(self, value):
        current = self._head
        foundvalue = False
        while current != None and not foundvalue:
            if current.getValue() == value:
                foundvalue = True
            else:
                current = current.getNext()
        return foundvalue

    # index索引元素在链表中的位置
    def index(self, value):
        current = self._head
        count = 0
        found = None
        while current != None and not found:
            count += 1
            if current.getValue() == value:
                found = True
            else:
                current = current.getNext()
        if found:
            return count
        else:
            raise ValueError('%s is not in linkedlist' % value)

    # remove删除链表中的某项元素
    def remove(self, value):
        current = self._head
        pre = None
        while current != None:
            if current.getValue() == value:
                if not pre:
                    self._head = current.getNext()
                else:
                    pre.setNext(current.getNext())
                break
            else:
                pre = current
                current = current.getNext()

    # insert链表中插入元素
    def insert(self, pos, value):
        if pos <= 1:
            self.add(value)
        elif pos > self.size():
            self.append(value)
        else:
            temp = Node(value)
            count = 1
            pre = None
            current = self._head
            while count < pos:
                count += 1
                pre = current
                current = current.getNext()
            pre.setNext(temp)
            temp.setNext(current)

    # 遍历链表
    def travel(self):
        current = self._head
        while current.getNext() != None:
            print(current.getValue())
            current = current.getNext()
        print(current.getValue())   #最后一位的next为空，所以也要打印最后一位的值


if __name__ == '__main__':
    lis = LinkedList()
    for i in range(10):
        lis.append(i)
    lis.append(11)
    print(lis.travel())
    print(lis._head.__dict__)
    print(lis._head.getNext().__dict__)
