创建型模式
-----

### 1\. 工厂模式（Factory Pattern）

**优势：**

*   封装了对象的创建过程，使代码更加灵活和可维护。
*   通过引入接口和抽象类，实现了松耦合，方便后续扩展和修改。
*   可以根据需要动态地创建不同类型的对象。

**适用场景：**

*   当需要根据条件创建不同对象时。
*   当对象的创建过程比较复杂，需要隐藏创建细节时。

**示例代码：**

```python
from abc import ABC, abstractmethod

class Product(ABC):
    @abstractmethod
    def operation(self):
        pass

class ConcreteProductA(Product):
    def operation(self):
        return "ConcreteProductA"

class ConcreteProductB(Product):
    def operation(self):
        return "ConcreteProductB"

class Creator(ABC):
    @abstractmethod
    def factory_method(self):
        pass

    def some_operation(self):
        product = self.factory_method()
        result = f"Creator: {product.operation()}"
        return result

class ConcreteCreatorA(Creator):
    def factory_method(self):
        return ConcreteProductA()

class ConcreteCreatorB(Creator):
    def factory_method(self):
        return ConcreteProductB()

# Client code
creator = ConcreteCreatorA()
result = creator.some_operation()
print(result)
```

### 2\. 单例模式（Singleton Pattern）

**优势：**

*   确保一个类只有一个实例，并提供一个全局访问点。
*   全局唯一实例可以避免资源重复分配和冲突。

**适用场景：**

*   当需要限制一个类只能有一个实例时。
*   当共享资源需要在整个应用程序中全局访问时。

**示例代码：**

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

# Client code
instance1 = Singleton()
instance2 = Singleton()

print(instance1 is instance2)  # Output: True
```

结构型模式
-----

### 1\. 适配器模式（Adapter Pattern）

**优势：**

*   允许不兼容接口的类一起工作。
*   提供了一种通过适配器类的转换机制来重用已有类的方式。

**适用场景：**

*   当需要将一个类的接口转换为另一个客户端所期望的接口时。
*   当需要复用已有类，但是其接口与应用要求不兼容时。

**示例代码：**

```python
class Target:
    def request(self):
        return "Target: The default target's behavior."

class Adaptee:
    def specific_request(self):
        return ".eetpadA eht fo roivaheb laicepS"

class Adapter(Target):
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def request(self):
        return f"Adapter: (TRANSLATED) {self.adaptee.specific_request()[::-1]}"

# Client code
adaptee = Adaptee()
adapter = Adapter(adaptee)
result = adapter.request()
print(result)
```

### 2\. 装饰器模式（Decorator Pattern）

**优势：**

*   动态地给对象添加额外的职责，而不修改其原始类。
*   可以通过组合多个装饰器来扩展对象的行为。

**适用场景：**

*   当需要在不修改现有对象结构的情况下，添加额外的功能或职责时。
*   当不适合使用子类进行扩展时，可以使用装饰器模式。

**示例代码：**

```python
from abc import ABC, abstractmethod

class Component(ABC):
    @abstractmethod
    def operation(self):
        pass

class ConcreteComponent(Component):
    def operation(self):
        return "ConcreteComponent"

class Decorator(Component):
    def __init__(self, component):
        self.component = component

    def operation(self):
        return f"Decorator({self.component.operation()})"

class ConcreteDecoratorA(Decorator):
    def operation(self):
        return f"ConcreteDecoratorA({self.component.operation()})"

class ConcreteDecoratorB(Decorator):
    def operation(self):
        return f"ConcreteDecoratorB({self.component.operation()})"

# Client code
component = ConcreteComponent()
decorator_a = ConcreteDecoratorA(component)
decorator_b = ConcreteDecoratorB(decorator_a)
result = decorator_b.operation()
print(result)
```

行为型模式
-----

### 1\. 观察者模式（Observer Pattern）

**优势：**

*   定义了一种一对多的依赖关系，使得多个观察者对象同时监听某一主题对象。
*   主题对象状态变化时，会通知所有观察者对象进行更新操作。

**适用场景：**

*   当一个对象的改变需要同时改变其他对象，并且不知道有多少对象需要改变时。
*   当一个对象必须通知其他对象，但又希望尽量减少它们之间的耦合时。

**示例代码：**

```python
class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update()

class ConcreteSubject(Subject):
    def __init__(self):
        super().__init__()
        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.notify()

class Observer:
    def update(self):
        pass

class ConcreteObserverA(Observer):
    def update(self):
        print("ConcreteObserverA: Reacted to the subject's event.")

class ConcreteObserverB(Observer):
    def update(self):
        print("ConcreteObserverB: Reacted to the subject's event.")

# Client code
subject = ConcreteSubject()
observer_a = ConcreteObserverA()
observer_b = ConcreteObserverB()

subject.attach(observer_a)
subject.attach(observer_b)

subject.state = "New State"
# Output:
# ConcreteObserverA: Reacted to the subject's event.
# ConcreteObserverB: Reacted to the subject's event.
```

### 2\. 策略模式（Strategy Pattern）

**优势：**

*   将可变行为封装为独立的策略类，使得算法可以独立于使用它的客户端变化。
*   提供了一种简单的扩展和切换算法的方式。

**适用场景：**

*   当需要在运行时选择算法的不同变体时。
*   当有多个相关的类仅在行为上有所不同时。

**示例代码：**

```python
from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def execute(self):
        pass

class ConcreteStrategyA(Strategy):
    def execute(self):
        return "ConcreteStrategyA"

class ConcreteStrategyB(Strategy):
    def execute(self):
        return "ConcreteStrategyB"

class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def execute_strategy(self):
        return self._strategy.execute()

# Client code
context = Context(ConcreteStrategyA())
result = context.execute_strategy()
print(result)

context.set_strategy(ConcreteStrategyB())
result = context.execute_strategy()
print(result)
```

### 3\. 命令模式（Command Pattern）

**优势：**

*   将请求封装为一个对象，使得可以用不同的请求对客户进行参数化。
*   支持撤销操作和事务的原子化执行。

**适用场景：**

*   当需要将请求发送者和请求接收者解耦时。
*   当需要支持请求的撤销、重做和事务的执行时。

**示例代码：**

```python
from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

class Receiver:
    def action(self):
        return "Receiver: Executing action."

class ConcreteCommand(Command):
    def __init__(self, receiver):
        self._receiver = receiver

    def execute(self):
        return self._receiver.action()

class Invoker:
    def set_command(self, command):
        self._command = command

    def execute_command(self):
        return self._command.execute()

# Client code
receiver = Receiver()
command = ConcreteCommand(receiver)
invoker = Invoker()
invoker.set_command(command)
result = invoker.execute_command()
print(result)
```

### 4\. 状态模式（State Pattern）

**优势：**

*   将对象的行为和状态分离，使得状态转换更加清晰和简单。
*   避免了大量的条件语句，使代码更加可维护和可扩展。

**适用场景：**

*   当一个对象的行为取决于它的状态，并且需要在运行时根据状态改变行为时。
*   当存在大量条件语句来判断对象的状态时，可以考虑使用状态模式进行重构。

**示例代码：**

```python
from abc import ABC, abstractmethod

class State(ABC):
    @abstractmethod
    def handle(self):
        pass

class ConcreteStateA(State):
    def handle(self):
        return "ConcreteStateA: Handling state A."

class ConcreteStateB(State):
    def handle(self):
        return "ConcreteStateB: Handling state B."

class Context:
    def __init__(self, state):
        self._state = state

    def change_state(self, state):
        self._state = state

    def request(self):
        return self._state.handle()

# Client code
state_a = ConcreteStateA()
state_b = ConcreteStateB()
context = Context(state_a)
result = context.request()
print(result)

context.change_state(state_b)
result = context.request()
print(result)
```

总结
--

以上是一些常见的设计模式示例，涵盖了创建型、结构型和行为型设计模式。设计模式为我们提供了一种解决软件设计问题的方法，可以提高代码的可读性、可维护性和可扩展性。在实际应用中，根据具体的问题和需求选择适合的设计模式，有助于构建高质量的软件系统。

请注意，设计模式的使用需要根据具体情况和需求进行权衡和决策。不是所有的问题都需要设计模式，而且滥用设计模式可能导致过度复杂化的代码。因此，在应用设计模式时，要确保模式的使用对解决问题是有益的，并且符合代码的可维护性和可扩展性的需求。