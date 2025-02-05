"""

Pytorch 提供自动计算梯度的功能，可以自动计算一个函数关于一个变量在某一取值下的导数，从而基于梯度对参数进行优化，这就是机器学习中的训练过程。使用 Pytorch 计算梯度非常容易，只需要执行 tensor.backward()，就会自动通过反向传播 (Back Propogation) 算法完成，后面我们在训练模型时就会用到该函数。

注意，为了计算一个函数关于某一变量的导数，Pytorch 要求显式地设置该变量是可求导的，即在张量生成时，设置 requires_grad=True。我们对上面计算
 的代码进行简单修改，就可以计算当
 时，

 和

 的值。
"""
import torch

x = torch.tensor([2.], requires_grad=True)
y = torch.tensor([3.], requires_grad=True)
z = (x + y) * (y - 2)
print(z)
# tensor([5.], grad_fn=<MulBackward0>)
z.backward()
print(x.grad, y.grad)
# tensor([1.]) tensor([6.])
