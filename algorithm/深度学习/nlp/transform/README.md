## 位置编码

位置编码通常是通过正弦和余弦函数来计算的，其公式如下：

PE(pos,2i)=sin⁡(pos100002i/dmodel)\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)PE(pos,2i)=sin(100002i/dmodel​pos​)PE(pos,2i+1)=cos⁡(pos100002i/dmodel)\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)PE(pos,2i+1)=cos(100002i/dmodel​pos​)

其中：

* PE(pos,2i)\text{PE}(pos, 2i)PE(pos,2i) 表示位置 pospospos 和维度 2i2i2i 的位置编码，
* PE(pos,2i+1)\text{PE}(pos, 2i+1)PE(pos,2i+1) 表示位置 pospospos 和维度 2i+12i+12i+1 的位置编码，
* pospospos 是位置的索引，
* iii 是维度的索引，i=0,1,2,…,dim_model−1i = 0, 1, 2, \ldots, \text{dim\_model} - 1i=0,1,2,…,dim_model−1，
* dim_model\text{dim\_model}dim_model 是词向量维度。

接下来，让我们用一个具体的例子来说明位置编码的计算过程。假设我们有一个长度为 5 的输入序列，词向量维度为 4。

我们先来计算位置 pos=0pos = 0pos=0 的位置编码。对于每个维度 iii，我们将应用正弦函数和余弦函数。

1. 对于 i=0i = 0i=0：PE(0,0)=sin⁡(0100000/4)=sin⁡(0)=0\text{PE}(0, 0) = \sin\left(\frac{0}{10000^{0/4}}\right) = \sin(0) = 0PE(0,0)=sin(100000/40​)=sin(0)=0PE(0,1)=cos⁡(0100000/4)=cos⁡(0)=1\text{PE}(0, 1) = \cos\left(\frac{0}{10000^{0/4}}\right) = \cos(0) = 1PE(0,1)=cos(100000/40​)=cos(0)=1
2. 对于 i=1i = 1i=1：PE(0,2)=sin⁡(0100002/4)=sin⁡(0)=0\text{PE}(0, 2) = \sin\left(\frac{0}{10000^{2/4}}\right) = \sin(0) = 0PE(0,2)=sin(100002/40​)=sin(0)=0PE(0,3)=cos⁡(0100002/4)=cos⁡(0)=1\text{PE}(0, 3) = \cos\left(\frac{0}{10000^{2/4}}\right) = \cos(0) = 1PE(0,3)=cos(100002/40​)=cos(0)=1

所以，位置 pos=0pos = 0pos=0 的位置编码为 [0,1,0,1][0, 1, 0, 1][0,1,0,1]。

同样的方法应用到其他位置上，直到我们计算完整个序列的位置编码为止。然后，这些位置编码将与词向量相加，得到最终的输入表示。