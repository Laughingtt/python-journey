#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： tian
# datetime： 2022/11/22 10:19 AM 
# ide： PyCharm
"""
例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回：

[3,9,20,15,7]


算法流程：
特例处理： 当树的根节点为空，则直接返回空列表 [] ；
初始化： 打印结果列表 res = [] ，包含根节点的队列 queue = [root] ；
BFS 循环： 当队列 queue 为空时跳出；
出队： 队首元素出队，记为 node；
打印： 将 node.val 添加至列表 tmp 尾部；
添加子节点： 若 node 的左（右）子节点不为空，则将左（右）子节点加入队列 queue ；
返回值： 返回打印结果列表 res 即可。

作者：Krahets
链接：https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/solutions/134956/mian-shi-ti-32-i-cong-shang-dao-xia-da-yin-er-ch-4/
来源：力扣（LeetCode）

BFS和DFS区别


BFS（广度优先搜索）和DFS（深度优先搜索）是两种常见的图遍历算法，它们在搜索图或树的过程中有不同的特点和应用场景。

BFS的基本思想是从图的某个起始点开始，依次访问其所有邻接节点，然后再访问这些邻接节点的邻接节点，以此类推，直到访问到所有可达的节点。具体实现时，可以使用队列来保存待访问的节点，保证先访问先出队。BFS通常用于寻找最短路径或最小步数的问题。

DFS的基本思想是从图的某个起始点开始，不断访问其未被访问的邻接节点，直到该节点的所有邻接节点都被访问完毕，然后返回到上一个节点继续访问其未被访问的邻接节点，直到遍历完整个图。具体实现时，可以使用递归或栈来保存待访问的节点，保证先访问后入栈或递归。DFS通常用于寻找所有可达路径的问题。

下面是BFS和DFS的具体区别：

访问顺序不同：BFS按照广度优先的顺序依次访问每个节点，而DFS按照深度优先的顺序先访问某个节点的所有子节点，然后再回溯到该节点的兄弟节点。

存储方式不同：BFS使用队列来保存待访问的节点，而DFS使用栈或递归来保存待访问的节点。

内存占用不同：BFS需要保存所有已访问节点的信息，因此可能需要更多的内存空间。而DFS只需要保存当前路径上的节点信息，因此内存占用通常较少。

适用场景不同：BFS通常用于寻找最短路径或最小步数的问题，而DFS通常用于寻找所有可达路径的问题。
"""
import collections
import queue


class TreeNode:
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right


class Solution:
    # 递归 深度优先
    def levelOrder(self, root: TreeNode):
        if root is None:
            return []
        tree_list = []

        def insert_tree_tree(level, sub_root, tree_lis):
            if level >= len(tree_lis):
                tree_lis.append([])
            tree_lis[level].append(sub_root.val)
            if sub_root.left is not None:
                insert_tree_tree(level + 1, sub_root.left, tree_lis)
            if sub_root.right is not None:
                insert_tree_tree(level + 1, sub_root.right, tree_lis)

        insert_tree_tree(0, root, tree_list)
        res = []
        for tree in tree_list:
            res += tree

        return res

    # BFS 广度优先搜索算法
    def levelOrder2(self, root: TreeNode):
        if root is None:
            return
        queue = collections.deque()
        queue.append(root)
        res_list = []
        while queue:
            node = queue.popleft()
            res_list.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        return res_list

    def bfs(self, root):
        res = []
        if root is None:
            return res

        q = queue.Queue()
        q.put(root)

        while q.qsize() != 0:
            node = q.get()
            res.append(node.val)

            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)
        return res



if __name__ == '__main__':
    s = Solution()
    tree = TreeNode(3, left=TreeNode(9, left=TreeNode(44, right=TreeNode(88))),
                    right=TreeNode(20, left=TreeNode(15), right=TreeNode(7)))
    print(s.levelOrder(tree))
    print(s.bfs(tree))
