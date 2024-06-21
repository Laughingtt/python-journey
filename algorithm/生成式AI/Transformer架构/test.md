\documentclass{article}
\usepackage{amsmath}

\begin{document}

\section*{位置编码公式（Positional Encoding）}

位置编码公式如下：

对于给定的位置 \( \text{pos} \) 和嵌入向量维度 \( i \)：

\begin{equation}
PE(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\end{equation}

\begin{equation}
PE(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\end{equation}

其中：
- \( \text{pos} \) 是位置索引。
- \( i \) 是嵌入向量的维度索引。
- \( d_{\text{model}} \) 是嵌入向量的维度。

\section*{示例说明}

假设嵌入向量的维度 \( d_{\text{model}} = 8 \)，我们计算位置 0 和位置 1 的部分位置编码：

\subsection*{位置 0 的位置编码}

对于位置 0 (\( \text{pos} = 0 \)) 和嵌入向量的第 \( 0 \) 和第 \( 1 \) 维度 (\( i = 0 \))：

\begin{equation}
PE(0, 0) = \sin\left(\frac{0}{10000^{\frac{0}{8}}}\right) = \sin(0) = 0
\end{equation}

\begin{equation}
PE(0, 1) = \cos\left(\frac{0}{10000^{\frac{0}{8}}}\right) = \cos(0) = 1
\end{equation}

\subsection*{位置 1 的位置编码}

对于位置 1 (\( \text{pos} = 1 \)) 和嵌入向量的第 \( 0 \) 和第 \( 1 \) 维度 (\( i = 0 \))：

\begin{equation}
PE(1, 0) = \sin\left(\frac{1}{10000^{\frac{0}{8}}}\right) = \sin(1)
\end{equation}

\begin{equation}
PE(1, 1) = \cos\left(\frac{1}{10000^{\frac{0}{8}}}\right) = \cos(1)
\end{equation}

\end{document}
