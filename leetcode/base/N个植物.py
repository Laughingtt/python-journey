"""
有N个植物，(1<= N = 20000)
列表 H记录了每个植物初始高度 (1到19 之间)
列表 A 记录了每个植物每一天生长的高度 (1到 19 之间)，同时有一个期望排名的
列表 T，其中 Ti是高度比第个植物高的其他植物的个数，T中N个元素的值范围为@到N-1 且没有重复。
给定 N、H、A和T，请输出需要达到目标排名的天数。如果目标不可能达成，请输出-1。
例如，对下面的例子，答案为4
N:5
H:7 4 1 10 12
A:3 4 5 2 1
T:2 1 0 3 4

7+12=19 4+16=20 20+1=21 8+10=18 4+12=16

2 1 0 3 4

对于下面的例子，答案为-1.
N:2
H: 7 3
A:8 6
T:1 0

比如N=10000，也能很快跑完。肯定要低于O(N)
"""
import copy


class Solution:
    def find_day(self, n_num, h_nums, a_nums, t_nums) -> int:
        """
        (h_nums + day * a_nums).index = t_nums
        """

        cur_nums = copy.deepcopy(h_nums)

        # 找出不可能超过种子
        for _t in t_nums:
            local_t_idx = t_nums.index(_t)
            for __t in t_nums:
                if _t > __t:
                    t_idx = t_nums.index(__t)
                    if h_nums[local_t_idx] >= h_nums[t_idx] and a_nums[local_t_idx] >= a_nums[t_idx]:
                        return -1

        # 找出合适的天数
        day = 1
        while True:
            for n in range(n_num):
                cur_nums[n] += a_nums[n]

            sort_cur = sorted(cur_nums, reverse=True)
            sort_idx = [sort_cur.index(_c) for _c in cur_nums]
            print(str(day) + " day " + str(cur_nums) + " index " + str(sort_idx))

            if sort_idx == t_nums:
                return day

            if day > 100:
                return -99
            day += 1

        # return 0

    @staticmethod
    def min_days_to_target_rank_dp(N, H, A, T):
        INF = float('inf')
        max_days = 10 ** 9  # 可以根据实际情况适当调整最大天数
        dp = [[INF] * (max_days + 1) for _ in range(N + 1)]

        # 初始状态，0天内都能达到目标排名
        for i in range(N + 1):
            dp[i][0] = 0

        for i in range(1, N + 1):
            for j in range(1, max_days + 1):
                # 计算在第j天内达到目标排名的最小天数
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1)

                # 模拟植物生长
                growth = H[i - 1] + j * A[i - 1]

                # 计算当前植物在第j天的排名
                rank = sum(1 for k in range(N) if growth > H[k] + j * A[k])

                # 更新最小天数
                dp[i][j] = min(dp[i][j], dp[i - 1][j], dp[i][rank] if rank < N else INF)

        result = min(dp[N])
        return result if result < INF else -1




if __name__ == '__main__':
    n_num = 5
    h_nums = [7, 4, 1, 10, 12]
    a_nums = [3, 4, 5, 2, 1]
    t_nums = [2, 1, 0, 3, 4]

    # n_num = 2
    # h_nums = [7, 3]
    # a_nums = [8, 6]
    # t_nums = [1, 0]
    s = Solution()
    # res = s.find_day(n_num, h_nums, a_nums, t_nums)
    res2 = s.min_days_to_target_rank_dp(n_num, h_nums, a_nums, t_nums)
    # print(res)
    print(res2)
