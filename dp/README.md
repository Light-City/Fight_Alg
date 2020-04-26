# DP专题

## 1.[300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

### 1.1 题目

题目：给定一个无序的整数数组，找到其中最长上升子序列的长度。

示例:

输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
说明:

可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
你算法的时间复杂度应该为 O(n2) 。
进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗?

### 1.2 解法

**明确以下两点**

- 不连续
- 严格上升

「子序列」和「子串」这两个名词的区别，子串一定是连续的，而子序列不一定是连续的

**状态**

- LIS(i)表示以第i哥数字为结尾的最长上升子序列的长度，也就是在[0...i]范围内，选择数字nums[i]结尾可以获得最长上升子序列的长度。
- 以第i个数字为结尾，也就是nums[i]肯定会被选择。

**状态转移方程**

- 以第i个数字为结尾，应该看看在[0...i-1]范围内的各个状态(LIS[j]，j表示0到i-1范围的某个数)，如果当前的数nums[i]大于[0...i-1]的**某些数**，那么nums[i]就可以跟刚才某些数的最大值进行拼接，组成更长的LIS，也就是LIS[i] = 某些数的最大LIS[j]加上1。所以状态转移方程为：LIS[i] = max(LIS[j]) + 1，约束条件是0<=j<=i-1，and nums[i] > nums[j]。

最终，最大值就是所有状态中最大的那个，也就是扫描LIS数组取最大。

（1）dp

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        
        int n = nums.size();
        
        if(n < 2) return n;

        vector<int> dp(n,1);
        int max_sum = 1;
        for(int i = 1; i < n; i++) {
            for(int j = i - 1; j >= 0; j--) {
                if(nums[i] > nums[j]) {
                    dp[i] = max(dp[i],dp[j]+1);
                    max_sum = max(max_sum, dp[i]);
                }
            }
        }

        return max_sum;
    }
};
```

上述优化：
```cpp
for(int j = i - 1; j >= 0; j--) {
    if(nums[i] > nums[j]) {
        dp[i] = max(dp[i],dp[j]+1);
    }
}
max_sum = max(max_sum, dp[i]);
```

（2）dp+二分查找法

> 参考自解图

扑克牌游戏(跟空当接龙有点像)，把大的放在下面，小的放在上面，从一堆无序的纸牌按照顺序依次取出扑克牌进行放置，例如：5 6 3 2 J。

第一堆：5 -> 3 -> 2

第二堆：6 

第三堆：J

可以发现最上层是有序的，堆数就是最终的结果。

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        
        int n = nums.size();
        
        if(n < 2) return n;
        vector<int> piles_top(n);  // 存储每个堆的堆顶元素
        // 纸牌堆数
        int piles = 0;
        for(int i=0;i<n;i++) {
            int poker = nums[i];
            int left=0, right = piles;

            // 查找当前元素在第几个堆上,边界二分查找
            while(left<right) {
                int mid = left + (right-left) / 2;
                if (piles_top[mid] > poker) {
                    right = mid;
                } else if (piles_top[mid] < poker){
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            // 创建新堆,或者说piles_top的元素个数
            if (left == piles) piles++;
            piles_top[left] = poker;
        }
        return piles;
        
    }
   
};
```

## 2.[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

### 2.1 题目

给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

若这两个字符串没有公共子序列，则返回 0。

### 2.2 解法

- 公共子序列易错点：

例如：text1 = "upr" , text2 = "urp"，那么公共子序列便是up 或者 ur，因此不能直接采用将一个子串的字符记录下来，然后判断另一个字符在不在记录里面，这种方式只能判断它在或者不在，并不能得到一个有序性的结果，因此结果不一定正确。

- 解法

下面分别为：自底向上的递归(递归1)与自顶向下的递归(递归2)，对应的便是自顶向下的dp(动态规化1)与自底向上的dp(动态规化2)。

要明白这道题到底是做什么，分析，再去做~

这道题是两个子序列求公共部分：例如下面两个子序列：

```
text1 = "abcde", text2 = "ace" 
```

我们可以从后往前，也可以从前往后，如果从递归角度，便对应上述的递归2与1。

我们以递归2(自顶向下)为例：上述例子中末尾e相等，往前计算，c与d不等，那么就会产生两种选择，text1向前移动，text2不动；text1不动，text2向前移动。

**总结：**

- 相等：都向前移动
- 不等：一个动，一个不动，取这两个结果的最大值，便是最终结果。

因此代码就很好写了，针对此引出记忆化搜索，存储text1在第i个位置，text2在第j个位置的时候，公共子序列长度为多少。进一步的，我们得到状态的定义：

- `dp[i][j]`表示在text1的第i个位置，text2在第j个位置，所保存的最长公共子序列长度。

状态转移方程：

- `text1[i] == text2[j]`
  - `dp[i][j] = 1 + dp[i-1][j-1]`

- `text1[i] != text2[j]`
  - `dp[i][j] = max(dp[i][j-1],dp[i-1][j])`

于是得到了动态规化的解法。

递归1：

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        return dfs(text1,text2,0,0);
    }

    int dfs(const string& text1,const string& text2,int l1,int l2) {
        if(l1 == text1.size() || l2 == text2.size()) {
            return 0;
        }

        if(text1[l1] == text2[l2]) 
            return 1 + dfs(text1,text2,l1+1,l2+1);
        return max(dfs(text1,text2,l1+1,l2), dfs(text1,text2,l1,l2+1));
    
    }
};
```

递归2：

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        return dfs(text1,text2,text1.size()-1,text2.size()-1);
    }

    int dfs(const string& text1,const string& text2,int l1,int l2) {
        if(l1 < 0 || l2 < 0) {
            return 0;
        }

        if(text1[l1] == text2[l2]) 
            return 1 + dfs(text1,text2,l1-1,l2-1);
        return max(dfs(text1,text2,l1-1,l2), dfs(text1,text2,l1,l2-1));
    
    }
};
```



记忆化搜索1：

```cpp
class Solution {
private:
    vector<vector<int>> memo;
public:
    int longestCommonSubsequence(string text1, string text2) {
        memo = vector<vector<int>>(text1.size(),vector<int>(text2.size(),-1));
        return dfs(text1,text2,0,0);
    }

    int dfs(const string& text1,const string& text2,int l1,int l2) {
        if(l1 == text1.size() || l2 == text2.size()) {
            return 0;
        }
        if(memo[l1][l2] != -1)
            return memo[l1][l2];

        if(text1[l1] == text2[l2]) 
            memo[l1][l2] = 1 + dfs(text1,text2,l1+1,l2+1);
        else 
            memo[l1][l2] = max(dfs(text1,text2,l1+1,l2), dfs(text1,text2,l1,l2+1));
        
        return memo[l1][l2];
    }
};
```

记忆化搜索2：

```cpp
class Solution {
private:
    vector<vector<int>> memo;
public:
    int longestCommonSubsequence(string text1, string text2) {
        memo = vector<vector<int>>(text1.size(),vector<int>(text2.size(),-1));
        return dfs(text1,text2,text1.size()-1,text2.size()-1);
    }

    int dfs(const string& text1,const string& text2,int l1,int l2) {
        if(l1 < 0 || l2 < 0) {
            return 0;
        }
        if(memo[l1][l2] != -1)
            return memo[l1][l2];

        if(text1[l1] == text2[l2]) 
            memo[l1][l2] = 1 + dfs(text1,text2,l1-1,l2-1);
        else 
            memo[l1][l2] = max(dfs(text1,text2,l1-1,l2), dfs(text1,text2,l1,l2-1));
        
        return memo[l1][l2];
    }
};
```

动态规化1：

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        vector<vector<int>> dp(text1.size() + 1,vector<int>(text2.size() + 1,0));

        // dp[0][j] 全部都等于0
        // dp[i][0] 全部都等于0

        for(int i=text1.size()-1; i>=0; i--) {
            for(int j=text2.size()-1; j>=0; j--) {
                if(text1[i] == text2[j]) {
                    dp[i][j] = 1 + dp[i+1][j+1];
                } else {
                    dp[i][j] = max(dp[i+1][j],dp[i][j+1]);
                }
            }
        }
        return dp[0][0];
    }
};
```

动态规化2：

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        vector<vector<int>> dp(text1.size() + 1,vector<int>(text2.size() + 1,0));

        // dp[0][j] 全部都等于0
        // dp[i][0] 全部都等于0

        for(int i=1; i<=text1.size();i++) {
            for(int j=1; j<=text2.size();j++) {
                if(text1[i-1] == text2[j-1]) {
                    dp[i][j] = 1 + dp[i-1][j-1];
                } else {
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[text1.size()][text2.size()];
    }
};
```

动态规化压缩：

由于在从左到右计算`dp[j]` 的时候`dp[j-1]`(`dp[i-1][j-1]`) 已被更新为`dp[j-1]`（`dp[i][j-1]`），所以只需要提前定义一个变量left_corn去存储二维dp数组左上方的值`dp[i-1][j-1]`,即未被更新前的`dp[j-1]`;

| `dp[i-1][j-1]` | `dp[i-1][j]` |
| -------------- | ------------ |
| `dp[i][j-1]`   | `dp[i][j]`   |

left_corn：存储左上角上次结果，也就是`dp[i-1][j-1]`。

up_tmp：存储正上方上次结果，也就是`dp[i-1][j]`。

dp[j-1]：相当于`dp[i][j-1]`

dp[j]：相当于`dp[i][j]`

实现如下：

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        vector<int> dp(text2.size()+1);
        int up_tmp;
        for(int i=1; i<=text1.size();i++) {
            int left_corn = 0;
            for(int j=1; j<=text2.size();j++) {
                up_tmp = dp[j];
                if(text1[i-1] == text2[j-1]) {
                    dp[j] = 1 + left_corn;  // dp[j-1]被覆盖了，所以取上次的
                } else {
                    dp[j] = max(dp[j],dp[j-1]);  
                }
                left_corn = up_tmp;
            }
        }
        return dp[text2.size()];
    }
};
```

## 3.[120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

### 3.1 题目

给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

例如，给定三角形：

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。

### 3.2 解法

- 坑！！！
  相邻本题指的是当前列index及列index+1，而不包括列index-1。

题目中说了，自顶向下，联想到递归与动态规划，先用递归吧，问啥子就写啥子呗。

> 解法1：递归法

```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    if(n == 0) return 0;
    int m = triangle[n-1].size();
    return dfs(triangle,0,0);   
}

int dfs(const vector<vector<int>>& triangle, int row, int col) {

    if(row == triangle.size()-1) return triangle[row][col];

    int cur = triangle[row][col] + dfs(triangle,row+1,col);
    int right = triangle[row][col] + dfs(triangle,row+1,col+1);
    int res = min(cur,right);
    return res;
}
```

> 解法2： 加memo 记忆化搜索


```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    if(n == 0) return 0;
    int m = triangle[n-1].size();

    memo = vector<vector<int>>(n,vector<int>(m,-1));
    return dfs(triangle,0,0);   
}

int dfs(const vector<vector<int>>& triangle, int row, int col) {

    if(row == triangle.size()-1) return triangle[row][col];


    if(memo[row][col] != -1) return memo[row][col];

    int cur = triangle[row][col] + dfs(triangle,row+1,col);
    int right = triangle[row][col] + dfs(triangle,row+1,col+1);

    int res = min(cur,right);
    memo[row][col] = res; 


    return memo[row][col];
}
```

> 解法3 自底向上，二维dp

```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    if(n == 0) return 0;
    int m = triangle[n-1].size();
    vector<vector<int>> dp(n+1,vector<int>(m+1,0));

    for(int i=n-1;i>=0;i--) {
        for(int j=0;j<triangle[i].size();j++) {
            dp[i][j] = triangle[i][j] + min(dp[i+1][j],dp[i+1][j+1]);
        }

    }

    return dp[0][0];   
}
```

> 一维dp：只保存前一行的最小值

```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    if(n == 0) return 0;
    int m = triangle[n-1].size();
    vector<int> dp(m+1,0);

    for(int i=n-1;i>=0;i--) {
        for(int j=0;j<triangle[i].size();j++) {
            dp[j] = triangle[i][j] + min(dp[j],dp[j+1]);
        }

    }

    return dp[0];   
}
```

## 4.[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

> 暴力法

```cpp
// 暴力解法
int maxSubArray1(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;

    int res = nums[0];
    for (int i = 0; i < n; i++)
    {

        int sum = 0;
        for (int j = i; j < n; j++)
        {
            sum += nums[j];
            res = max(res, sum);
        }
    }
    return res;
}
```

> 动态规划法

- 状态
  - dp[i]定义为数组nums中以num[i] 结尾的最大连续子串和
- 状态转移方程
  - dp[i] = max(dp[i-1]+nums[i],nums[i])
- 状态压缩 -> 优化数组空间
  - 每次状态的更新只依赖于前一个状态，见后一个方法
- 选出结果
  - 有的题目结果是 dp[i] 。
  - 本题结果是 dp[0]...dp[i] 中最大值。


```cpp
// 动态规划法
// 状态 ： dp[i]定义为数组nums中以num[i] 结尾的最大连续子串和
// 状态转移方程 : dp[i] = max(dp[i-1]+nums[i],nums[i])
int maxSubArray2(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    vector<int> dp(n);
    dp[0] = nums[0];
    int res = dp[0];
    for (int i = 1; i < n; i++)
    {
        dp[i] = max(dp[i - 1] + nums[i], nums[i]);
        res = max(res, dp[i]);
    }
    return res;
}
```

> 考虑到当前状态只依赖于前面一个状态，可以压缩为常数空间

```cpp
// 上述状态压缩
int maxSubArray3(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    int mx = nums[0];
    int res = mx;
    for (int i = 1; i < n; i++)
    {
        mx = max(mx + nums[i], nums[i]);
        res = max(res, mx);
    }
    return res;
}
```

> 贪心法

每次当累计sum大于0，继续往后计算，否则取当前元素。

```cpp
// 贪心法
int maxSubArray4(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    int sum = 0;
    int res = nums[0];
    for (auto num : nums)
    {
        sum = sum > 0 ? sum + num : num;
        res = max(res, sum);
    }
    return res;
}
// 参考题解 得出的类似归并的分治算法
int maxSubArray5(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    int res = nums[0];
    res = divide_alg(nums, 0, n - 1);

    return res;
}
```

> 分治法

数组 [-2,1,-3,4,-1,2,1,-5,4] ，一共有 9 个元素，我们 mid=(left + right) / 2 这个原则，得到中间元素的索引为 4 ，也就是 -1，拆分成三个组合：

- [-2,1,-3,4,-1]以及它的子序列（在-1左边的并且包含它的为一组）
- [2,1,-5,4]以及它的子序列（在-1右边不包含它的为一组）
- 任何包含-1以及它右边元素2的序列为一组（换言之就是包含左边序列的最右边元素以及右边序列最左边元素的序列，比如 [4,-1,2,1]，这样就保证这个组合里面的任何序列都不会和上面两个重复）

以上的三个组合内的序列没有任何的重复的部分，而且一起构成所有子序列的全集，计算出这个三个子集合的最大值，然后取其中的最大值，就是这个问题的答案了。

然而前两个子组合可以用递归来解决，一个函数就搞定，第三个跨中心的组合应该怎么计算最大值呢？

答案就是**先计算左边序列里面的包含最右边元素的子序列的最大值，也就是从左边序列的最右边元素向左一个一个累加起来，找出累加过程中每次累加的最大值，就是左边序列的最大值。 同理找出右边序列的最大值，就得到了右边子序列的最大值。左右两边的最大值相加，就是包含这两个元素的子序列的最大值。**

在计算过程中，累加和比较的过程是关键操作，一个长度为 n 的数组在递归的每一层都会进行 n 次操作，分治法的递归层级在 logN 级别，所以整体的时间复杂度是 O(nlogn)，在时间效率上不如动态规划的 O(n)复杂度。

连续子序列的最大和主要由这三部分子区间里元素的最大和得到：

- 第 1 部分：子区间 [left, mid]；
- 第 2 部分：子区间 [mid + 1, right]；
- 第 3 部分：包含子区间[mid , mid + 1]的子区间，即 nums[mid] 与nums[mid + 1]一定会被选取。

参考自：

> https://leetcode-cn.com/problems/maximum-subarray/solution/zheng-li-yi-xia-kan-de-dong-de-da-an-by-lizhiqiang/

实现如下：

```cpp
// 参考题解 得出的类似归并的分治算法
int maxSubArray5(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    int res = nums[0];
    res = divide_alg(nums, 0, n - 1);

    return res;
}
int divide_alg(const vector<int> &nums, int left, int right)
{
    // 只有一个元素
    if (left == right)
    {
        return nums[left];
    }
    int mid = left + (right - left) / 2;
    int left_sum = divide_alg(nums, left, mid);
    int right_sum = divide_alg(nums, mid + 1, right);
    int left_cross_sum = nums[mid];
    int sum = 0;
    for (int i = mid; i >= left; i--)
    {
        sum += nums[i];
        left_cross_sum = max(left_cross_sum, sum);
    }

    int right_cross_sum = nums[mid + 1];
    sum = 0;
    for (int i = mid + 1; i <= right; i++)
    {
        sum += nums[i];
        right_cross_sum = max(right_cross_sum, sum);
    }
    int res = max(max(left_sum, right_sum), left_cross_sum + right_cross_sum);
    return res;
}
```

## 5.[152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

本题同第4题思路基本一致，直接开始写思路了。

> 阐述思路


考虑如下情形：

- nums[i]>0 此时最大乘积为之前的结果，记作dp_max[i-1]
  - 若dp_max[i-1]>0: dp_max[i] = dp_max[i-1]*nums[i]
  - 若dp_max[i-1]<=0: dp_max[i] = nums[i]
- nums[i]<=0 那么此时最大乘积就是：dp_max[i] = 之前的最小值*nums[i] 或者 nums[i]
  - 若dp_min[i-1]<=0: dp_max[i] = dp_min[i-1]*nums[i]
  - 若dp_min[i-1]>0: dp_max[i] = nums[i]

小结:

- 一个dp解决不了问题，需要两个dp，既要维护最大值，也要维护最小值

于是因此二维dp或者两个dp:

- 状态
  - dp[0][i] 存储以第 i 个数结尾的 乘积最小 的连续子序列值
  - dp[1][i] 存储以第 i 个数结尾的 乘积最大 的连续子序列值
- 状态转移方程
  - dp[0][i] = max(dp[0][i-1]*nums[i],nums[i])
  - dp[1][i] = min(dp[1][i-1]*nums[i],nums[i])

> 二维动态规划

```cpp
int maxProduct(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    // dp[0][i] 存储以第 i 个数结尾的 乘积最小 的连续子序列值
    // dp[1][i] 存储以第 i 个数结尾的 乘积最大 的连续子序列值
    vector<vector<int>> dp(2, vector<int>(n + 1, 1));
    int res = nums[0];
    for (int i = 1; i <= n; i++)
    {
        if (nums[i - 1] < 0)
            swap(dp[0][i - 1], dp[1][i - 1]);
        dp[0][i] = min(nums[i - 1], dp[0][i - 1] * nums[i - 1]);
        dp[1][i] = max(nums[i - 1], dp[1][i - 1] * nums[i - 1]);
        res = max(res, dp[1][i]);
    }
    return res;
}
```

> 两个一维dp

```cpp
int maxProduct(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    // dp_min[i] 存储以第 i 个数结尾的 乘积最小 的连续子序列值
    vector<int> dp_min(n + 1, 1);
    // dp_max[i] 存储以第 i 个数结尾的 乘积最大 的连续子序列值
    vector<int> dp_max(n + 1, 1);
    int res = nums[0];
    for (int i = 1; i <= n; i++)
    {
        if (nums[i - 1] < 0)
            swap(dp_min[i - 1], dp_max[i - 1]);
        dp_min[i] = min(nums[i - 1], dp_min[i - 1] * nums[i - 1]);
        dp_max[i] = max(nums[i - 1], dp_max[i - 1] * nums[i - 1]);
        res = max(res, dp_max[i]);
    }
    return res;
}
```

> 状态压缩方法1

根据上述的阐述我们可以得出如下两种结果：

- 状态压缩：每次只与前面一个状态有关
- dp最大值与最小值，始终都是在`dpMax*nums[i]`,`*dpMin*nums[i]`,`nums[i]`三者中取最大与最小。

```cpp
int maxProduct(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    int dpMax = nums[0];
    int dpMin = nums[0];
    int res = nums[0];
    for (int i = 1; i < n; i++)
    {
        //更新 dpMin 的时候需要 dpMax 之前的信息，所以先保存起来
        int preMax = dpMax;
        dpMax = max(max(dpMin * nums[i],dpMax * nums[i]), nums[i]);
        dpMin = min(min(dpMin * nums[i],preMax * nums[i]), nums[i]);
        res = max(res, dpMax);
    }
    return res;
}
> 状态压缩方法二

直接按照dp二维压缩
​```cpp
int maxProduct(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    int res = nums[0];
    int dp_min = 1;
    int dp_max = 1;
    for (int i = 1; i <= n; i++)
    {
        if (nums[i - 1] < 0)
            swap(dp_min, dp_max);
        dp_min = min(nums[i - 1], dp_min * nums[i - 1]);
        dp_max = max(nums[i - 1], dp_max * nums[i - 1]);
        res = max(res, dp_max);
    }
    return res;
}
```

> 题解中的方案

参考自：

> https://leetcode-cn.com/problems/maximum-product-subarray/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by--36/

偶数个负数,乘积最大为所有元素
奇数个负数,乘积最大

- 不包含第一个负数
- 不包含第二个负数

如果有 0 存在的话，会使得上边的代码到 0 的位置之后 max 就一直变成 0 了。

- 只需要将tmp设置为1，重新往后计算即可。

```cpp
int maxProduct(vector<int> &nums)
{
    int n = nums.size();
    if (n == 0)
        return 0;
    int ret = 1;
    int res = nums[0];
    for (int i = 0; i < nums.size(); i++)
    {
        ret *= nums[i];
        res = max(res, ret);
        if (nums[i] == 0)
        {
            ret = 1;
        }
    }
    ret = 1;
    for (int i = nums.size() - 1; i >= 0; i--)
    {
        ret *= nums[i];
        res = max(res, ret);
        if (nums[i] == 0)
        {
            ret = 1;
        }
    }
    return res;
}
```

## 6.[887. 鸡蛋掉落](https://leetcode-cn.com/problems/super-egg-drop/)

题目：
你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。

每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。

你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，从 F 楼层或比它低的楼层落下的鸡蛋都不会破。

每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。

你的目标是确切地知道 F 的值是多少。

无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？


- 题目简化

找摔不碎鸡蛋的最高楼层 F，「最坏情况」下「至少」要扔几次。

例如：6层楼，1个鸡蛋，一层层试，最坏情况下到了第6层也不碎，也就是移动了6次。


- 状态: 当前拥有的鸡蛋数 K 和需要测试的楼层数 N。随着测试的进行，鸡蛋个数可能减少，楼层的搜索范围会减小，这就是状态的变化。


当我们选择在第 i 层楼扔了鸡蛋之后，可能出现两种情况：鸡蛋碎了，鸡蛋没碎。注意，这时候状态转移就来了：

如果鸡蛋碎了，那么鸡蛋的个数 K 应该减一，搜索的楼层区间应该从 [1..N] 变为 [1..i-1] 共 i-1 层楼；

如果鸡蛋没碎，那么鸡蛋的个数 K 不变，搜索的楼层区间应该从 [1..N] 变为 [i+1..N] 共 N-i 层楼。


- 状态转移: 
  - 如果鸡蛋碎了，那么鸡蛋的个数 K 应该减一，搜索的楼层区间应该从 [1..N] 变为 [1..i-1] 共 i-1 层楼；

  - 如果鸡蛋没碎，那么鸡蛋的个数 K 不变，搜索的楼层区间应该从 [1..N] 变为 [i+1..N] 共 N-i 层楼。


因此得出：dp[K][N] = 1 + min(dp[k][N-1],dp[K-1][N-i])


实现如下：

```cpp
class Solution {
private:
    vector<vector<int>> memo;
public:
    int superEggDrop(int K, int N) {
        memo = vector<vector<int>>(K+1,vector<int>(N+1,-1));
        return dp(K,N);
    }

    int dp(int K, int N) {
        if(K==1) return N;
        if(N==0) return 0;

        if(memo[K][N] != -1) return memo[K][N];
        // 状态：当前拥有的鸡蛋数K和需要测试的楼层数N。
        // 状态转移dp[K][i] = 1 + dp[K-1][i-1] + dp[K][N-i]
        int res = INT_MAX;
        for(int i=1;i<=N;i++) {
            res = min(res,1+max(dp(K-1,i-1),dp(K,N-i)));
        }
        memo[K][N] = res;
        return res;
    }
};
```

> 二分查找优化

```cpp
class Solution
{
private:
    vector<vector<int>> memo;

public:
    int superEggDrop(int K, int N)
    {
        memo = vector<vector<int>>(K + 1, vector<int>(N + 1, -1));
        return dp(K, N);
    }

    int dp(int K, int N)
    {
        if (K == 1)
            return N;
        if (N == 0)
            return 0;

        if (memo[K][N] != -1)
            return memo[K][N];
        // 状态：当前拥有的鸡蛋数K和需要测试的楼层数N。
        // 状态转移dp[K][i] = 1 + dp[K-1][i-1] + dp[K][N-i]
        int res = INT_MAX;

        int l = 1, h = N;
        while (l <= h)
        {
            int mid = l + (h - l) / 2;
            // 碎
            int b = dp(K - 1, mid - 1);
            // 没碎
            int no_b = dp(K, N - mid);
            if (b > no_b)
            {
                h = mid - 1;
                res = min(res, b + 1);
            }
            else
            {
                l = mid + 1;
                res = min(res, no_b + 1);
            }
        }

        memo[K][N] = res;
        return res;
    }
};

```

> 自底向上

```cpp
class Solution {
private:
    vector<int> memo;
public:
    int superEggDrop(int K, int N) {
        vector<vector<int> > dp(K + 1, vector<int>(N+1, 0));
        // K个鸡蛋扔在第一层楼
        for (int i = 1; i <= K; i++) dp[i][1] = 1;
        // 1个鸡蛋测试N层楼
        for (int j = 1; j <= N; j++) dp[1][j] = j;
        
        for (int i = 2; i <= K; i++) {
            for (int j = 2; j <= N; j++) {
                // i个鸡蛋 j层楼所需要移动的次数
                // 首先第一个鸡蛋仍在t层楼 则 碎或者不碎 二分 max(f(K-1,t-1), f(K,N-t)) + 1
                int res = INT_MAX;
                
                int l = 1, h = j;
                while (l <= h)
                {
                    int mid = l + (h - l) / 2;
                    // 碎
                    int b = dp[i - 1][mid - 1];
                    // 没碎
                    int no_b = dp[i][j-mid];
                    if (b > no_b)
                    {
                        h = mid - 1;
                        res = min(res, b + 1);
                    }
                    else
                    {
                        l = mid + 1;
                        res = min(res, no_b + 1);
                    }
                }
                dp[i][j] = res;
            }
        }
        return dp[K][N];
    }
};
```




> 重新定义状态转移

- dp[k][n] 当前状态为 k 个鸡蛋，面对 n 层楼 返回这个状态下最少的扔鸡蛋次数
  - 鸡蛋碎： 剩下j-1个鸡蛋,i-1次操作,dp[i-1][j-1]
  - 鸡蛋不碎：剩下j个鸡蛋,i-1次操作,dp[i-1][j]


实现：

```cpp
class Solution {
private:
   
public:
    int superEggDrop(int K, int N) {
        vector<vector<int>> dp(K+1,vector<int>(N+1,0));

        int m = 0;
        // dp[k][n] 当前状态为 k 个鸡蛋，面对 n 层楼 返回这个状态下最少的扔鸡蛋次数

        // 鸡蛋碎： 剩下j-1个鸡蛋,i-1次操作,dp[i-1][j-1]
        // 鸡蛋不碎：剩下j个鸡蛋,i-1次操作,dp[i-1][j]

        while(dp[K][m]<N) {
            ++m;
            for(int k=1;k<=K;k++) {
                dp[k][m] = dp[k][m-1] + dp[k-1][m-1] + 1;
            }
        }
        return m;
    }
};
```
## 7.[354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)

本题是一道典型的LIS，套路一毛一样，只需要先排序，后面完全按照第一题做法就行了。

> 二维dp

```cpp
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        int n = envelopes.size();
        if(n<2) return n;
        
        sort(envelopes.begin(), envelopes.end(), cmp);
        // [2,3] [5,4] [6,4] [6,7]
        
        vector<int> dp(n,1);
        int max_sum = 1;
        for(int i=1;i<n;i++) {
            for(int j=i-1;j>=0;j--) {
                if(envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1]) {
                    dp[i] = max(dp[i],dp[j]+1);
                }
            }
            max_sum = max(max_sum,dp[i]);
        }
        
        return max_sum;
        
    }
    
    static bool cmp(const vector<int> &a, const vector<int> &b)
    {
        return a[0] < b[0];
    }
};
```
---
## 打家劫舍系列

## 8.[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

### 8.1 记忆化搜索

每次有两种选择，选与不选。

```cpp
class Solution {
private:
    vector<int> memo;
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        memo = vector<int>(n,-1);
        return dfs(nums,n-1);
    }


    int dfs(const vector<int>& nums,int index) {
        if(index<0) return 0;

        if(memo[index]!=-1) {
            return memo[index];
        }

        int res = 0;
        
        // 选择与不选择
        res = max(nums[index]+dfs(nums,index-2),dfs(nums,index-1));
    
        memo[index] = res;
        return res;
    }
};
```
当然去掉记忆化搜索就是动态规划。

### 8.2 动态规划
- 状态
    - dp[i] 保存每次偷盗第i个房子时，所偷的最多金额
- 状态转移方程
    - dp[i] = max(dp[i-1],nums[i]+dp[i-2]);
    
```cpp
class Solution {

public:
    int rob(vector<int>& nums) {
        int n = nums.size();

        if(n==0) return 0;
        if(n==1) return nums[0];        
        vector<int> dp(n,0);

        dp[0] = nums[0];
        dp[1] = max(nums[0],nums[1]);
        for(int i=2;i<n;i++) {
            dp[i] = max(dp[i-1],nums[i]+dp[i-2]);
        }
        
        return dp[n-1];
    }
};
```

### 8.3 滚动数组

```cpp
class Solution {

public:
    int rob(vector<int>& nums) {
        int n = nums.size();

        if(n==0) return 0;
        if(n==1) return nums[0];        

        vector<int> dp(3,0);
        dp[0] = nums[0];
        dp[1] = max(nums[0],nums[1]);

        for(int i=2;i<n;i++) {
            dp[i%3] = max(dp[(i-1)%3],nums[i]+dp[(i-2)%3]);
        }
        
        return max(max(dp[0],dp[1]),dp[2]);
    }
};
```
### 8.4 状态压缩

```cpp
class Solution {

public:
    int rob(vector<int>& nums) {
        int n = nums.size();

        if(n==0) return 0;
        if(n==1) return nums[0];        

        int pre = 0;
        int pre_pre = 0;
        
        int res;
        for(int i=0;i<n;i++) {
            res = max(pre,nums[i]+pre_pre);
            pre_pre = pre;
            pre = res;
        }
        
        return res;
    }
};
```
## 9.[213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

### 9.1 记忆化搜索

```cpp
class Solution {
private:
    vector<int> memo;
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        memo = vector<int>(n,-1);
        
        if(n==0) return 0;
        if(n==1) return nums[0];
        if(n==2) return max(nums[0],nums[1]);

        int b = nums.back();
        nums.pop_back();
        int r_1 = dfs(nums,n-2);

        memo = vector<int>(n,-1);
        nums.push_back(b);
        nums.erase(nums.begin());
        int r_2 = dfs(nums,n-2);
        

        return max(r_1,r_2);
    }


     int dfs(const vector<int>& nums,int index) {
        if(index<0) return 0;

        if(memo[index]!=-1) {
            return memo[index];
        }

        int res = 0;
        
        // 选择与不选择
        res = max(nums[index]+dfs(nums,index-2),dfs(nums,index-1));
    
        memo[index] = res;
        return res;
    }
};
```
上述优化版：
```cpp
class Solution {
private:
    vector<int> memo;
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        memo = vector<int>(n,-1);
        
        if(n==0) return 0;
        if(n==1) return nums[0];
        if(n==2) return max(nums[0],nums[1]);


        int r1 = dfs(nums,0,n-2);
        // 清理第一次结果
        memo = vector<int>(n,-1);
        int r2 = dfs(nums,1,n-1);
        return max(r1,r2);
    }


     int dfs(const vector<int>& nums,int start, int index) {
        if(index<start) return 0;

        if(memo[index]!=-1) {
            return memo[index];
        }

        int res = 0;
        
        // 选择与不选择
        res = max(nums[index]+dfs(nums,start,index-2),dfs(nums,start,index-1));
    
        memo[index] = res;
        return res;
    }
};
```

### 9.2 动态规划法

```cpp
class Solution {
public:
        int rob(vector<int>& nums) {
        int n = nums.size();

        if(n==0) return 0;
        if(n==1) return nums[0];        

        vector<int> dp1(nums.size(),0);
        vector<int> dp2(nums.size(),0); 
        // 0~n-2  
        dp1[0] = nums[0];
        dp1[1] = max(nums[0],nums[1]);

        for(int i=2;i<nums.size()-1;i++) {
            dp1[i] = max(dp1[i-1],nums[i]+dp1[i-2]);
        }

        // 1~n-1
        dp2[0]=0;
        dp2[1]=nums[1];
        for(int i=2;i<nums.size();i++) {
            dp2[i] = max(dp2[i-1],nums[i]+dp2[i-2]);
        }

        return max(dp1[n-2],dp2[n-1]);
    }
};
```
### 9.3 状态压缩

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if(n==0) return 0;
        if (n == 1) return nums[0];
        return max(__rob(nums,0,n-2),__rob(nums,1,n-1));
    }


     int __rob(const vector<int>& nums,int start, int end) {
        
        int pre = 0;
        int pre_pre = 0;
        int res;
        for(int i=start;i<=end;i++) {
            res = max(pre,nums[i]+pre_pre);
            pre_pre = pre;
            pre = res;
        }

        return res;
    }
};
```
## 10 [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)


### 10.1 记忆化搜索

每次抢劫，分为两种情况：
- 选择
    - 当前孩子及抢两个孩子的孩子们
- 不选择
    - 抢两个孩子之和
状态转移：dp[root] = max(当前孩子及两个孩子的孩子们，两个孩子之和);

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {

private:

    unordered_map<TreeNode*,int> memo;
public:
    int rob(TreeNode* root) {
        return dfs(root);
    }


    int dfs(TreeNode* root) {

        if(!root) {
            return 0;
        }
        if(memo.count(root))
            return memo[root];
 
        // 选择当前节点 -> 偷左右子节点的左右子节点
        int res1 = root->val;
        
        if(root->left) {
            res1 += (dfs(root->left->left)+dfs(root->left->right));
        }
        if(root->right) {
            res1 += (dfs(root->right->left)+dfs(root->right->right));
        }

        // 不选择
        int res2 = dfs(root->left) + dfs(root->right);
        
        memo[root]=max(res1,res2);
        return memo[root];
    }
};
```

### 10.2 动态规划

上述思想简化:
每次都分为两种，抢与不抢，或者说选择与不选择。
那么我们可以用两个状态来记录当前抢了还是没抢。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */


class Solution {
private:
    struct Rob {
        int rob;
        int no_rob;
    };
public:
    int rob(TreeNode* root) {
        Rob r = dp(root);
        return max(r.rob,r.no_rob);
    }


    Rob dp(TreeNode* root) {
        Rob r;
        if(!root) return r;
        Rob rl = dp(root->left);
        Rob rr = dp(root->right); 
        // 抢 当前抢，孩子肯定不抢
        int r_rob = root->val + rl.no_rob + rr.no_rob;

        // 不抢 当前不抢，孩子可抢，可不抢，取决于收益大小。 取两个孩子抢与不抢最大值之和
        int not_r_rob = max(rl.no_rob,rl.rob) + max(rr.no_rob,rr.rob);

        r.rob = r_rob;
        r.no_rob = not_r_rob;
        return r;
    }
};
```
---
## 买卖股票系列

## 11.[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

> 一块块计算，分块求最大

```cpp
// 一块块计算，分块求最大
int maxProfit(vector<int> &prices)
{
    int n = prices.size();
    if (n < 2)
        return 0;
    int res = 0;
    int min_val = prices[0];

    for (int i = 1; i < n; i++)
    {
        res = max(res, prices[i] - min_val);
        min_val = min(min_val, prices[i]);
    }
    return res;
}
```
> 动态规划

```cpp
// 要注意：一开始不持股与卖出后不持股的区别。因为涉及到买卖次数的问题。
// dp[i][0] 表示第i天未持有股票所获得的最大利润
// 状态转移： 第一种是第i天保留之前状态，另一种是卖股票(那就是之前的买入状态转移到现在状态)
// dp[i][1] 表示第i天持有股票所获得的最大利润
// 状态转移： 第一种是之前仍旧持有状态保留、第二种是之前卖出到买入的转移 而k=1的情况下 前一次卖出一定为0
// -prices[i]：注意：状态 1 不能由状态 0 来，因为事实上，状态 0 特指：“卖出股票以后不持有股票的状态”，请注意这个状态和“没有进行过任何一次交易的不持有股票的状态”的区别。
// 因此，-prices[i] 就表示，在索引为 i 的这一天，执行买入操作得到的收益。注意：因为题目只允许一次交易，因此不能加上 dp[i - 1][0]。
int maxProfit(vector<int> &prices)
{
    int n = prices.size();
    vector<vector<int>> dp(n + 1, vector<int>(2, 0));
    if (n == 0)
        return 0;
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    for (int i = 1; i < n; i++)
    {
        // 不持股票
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
        // 持有股票
        dp[i][1] = max(dp[i - 1][1], -prices[i]);
    }
    return dp[n - 1][0];
}
```
> 状态压缩

```cpp
int maxProfit(vector<int> &prices)
{

    int n = prices.size();
    if (n == 0)
        return 0;
    int profit_0 = 0;
    int profit_1 = -prices[0];
    for (int i = 1; i < n; i++)
    {
        // 不持股票
        profit_0 = max(profit_0, profit_1 + prices[i]);
        // 持有股票
        profit_1 = max(profit_1, -prices[i]);
    }
    return profit_0;
}

```

## 12 [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)


> 记忆化搜索

```cpp
// 记忆化搜索
private:
vector<vector<int>> memo;

public:
int maxProfit3(vector<int> &prices)
{
    int n = prices.size();
    memo = vector<vector<int>>(n, vector<int>(2, -1));

    // true表示持有股票
    // false表示未持有股票
    return dfs(prices, 0, 0);
}
int dfs(const vector<int> &prices, int index, int has_stock)
{
    if (index == prices.size())
        return 0;

    if (memo[index][has_stock] != -1)
        return memo[index][has_stock];

    int res = 0;
    if (has_stock)
    {
        // 选择卖
        res = dfs(prices, index + 1, 0) + prices[index];
    }
    else
    {
        // 选择买
        res = dfs(prices, index + 1, 1) - prices[index];
    }
    // 不操作
    int cur_retain = dfs(prices, index + 1, has_stock);
    memo[index][has_stock] = max(res, cur_retain);
    return memo[index][has_stock];
}
```
> 动态规划

```cpp
int maxProfit(vector<int> &prices)
{
    int n = prices.size();
    vector<vector<int>> dp(n, vector<int>(2));

    if (n == 0)
        return 0;

    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    for (int i = 1; i < n; i++)
    {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]); // 未持有
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]); // 持有
    }
    return dp[n - 1][0];
}
```
> 状态压缩

```cpp
int maxProfit1(vector<int> &prices)
{
    int n = prices.size();

    if (n == 0)
        return 0;
    int profit_0 = 0;
    int profit_1 = -prices[0];
    for (int i = 1; i < n; i++)
    {
        profit_0 = max(profit_0, profit_1 + prices[i]); // 未持有
        profit_1 = max(profit_1, profit_0 - prices[i]); // 持有
    }
    return profit_0;
}
```

> 贪心算法

```cpp
// 贪心算法 每一步都记录了当前最优选择  
// 贪心算法和动态规划相比，它既不看前面（也就是说它不需要从前面的状态转移过来），也不看后面（无后效性，后面的选择不会对前面的选择有影响），
// 因此贪心算法时间复杂度一般是线性的，空间复杂度是常数级别的。
// 在众多策略中选择一个最优的，如本题：每次会有三种情况：0,负数，正数
// 所以求最大利润，肯定选择正数即可
int maxProfit2(vector<int> &prices)
{
    int n = prices.size();

    int res = 0;
    for (int i = 1; i < n; i++)
    {
        int sub_res = prices[i] - prices[i - 1];
        // 只在底谷买入的时候才计算
        if (sub_res > 0)
            res += sub_res;
    }
    return res;
}
```

## 13.[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

> 记忆化搜索

```cpp
class Solution {
    // 记忆化搜索
private:
    vector<vector<vector<int>>> memo;

public:
    int maxProfit(vector<int> &prices)
    {
        int n = prices.size();
        // 因为股票买卖也可以进行0次，所以有0,1,2三种
        memo = vector<vector<vector<int>>>(n, vector<vector<int>>(3, vector<int>(2, -1)));
        return dfs(prices, 0, 0, 0);
    }
    int dfs(const vector<int> &prices, int index, int has_stock, int counts)
    {
        // 未持有股票且买卖超过2次，结束
        if (index == prices.size() || (counts == 2 && has_stock == 0))
        {
            return 0;
        }

        if (memo[index][counts][has_stock] != -1)
            return memo[index][counts][has_stock];

        int res = 0;
        if (has_stock)
        {
            // 选择卖
            res = dfs(prices, index + 1, 0, counts) + prices[index];
        }
        else
        {
            // 选择买
            res = dfs(prices, index + 1, 1, counts + 1) - prices[index];
        }
        // 不操作
        int cur_retain = dfs(prices, index + 1, has_stock, counts);
        memo[index][counts][has_stock] = max(res, cur_retain);
        return memo[index][counts][has_stock];
    }
};
```
> 动态规划法

```
int maxProfit(vector<int> &prices)
{
    int n = prices.size();
    if (n == 0)
        return 0;
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(3, vector<int>(2)));

    // dp[i][k][0] 表示第i个股票时候未持有股票所获得的最大利润
    // dp[i][k][1] 表示第i个股票时候持有股票所获得的最大利润

    for (int i = 0; i < n; i++)
    {
        for (int k = 1; k <= 2; k++)
        {
            if(i==0) {
                dp[i][k][0] = 0;
                dp[i][k][1] = -prices[i];
                continue;
            }
            dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
            // 当前持有股票 {原先持有，买入新股票}
            dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k-1][0] - prices[i]);
        }
    }
    return dp[n - 1][2][0];
}
```

> 状态压缩

```cpp
int maxProfit(vector<int> &prices)
{
    int n = prices.size();
    if (n == 0)
        return 0;


    // dp_i10 第一次未持有股票可获得的最大利润
    // dp_i11 第一次持有股票可获得的最大利润
    // dp_i20 第二次未持有股票可获得的最大利润
    // dp_i21 第二次持有股票可获得的最大利润
    int dp_i10 = 0,dp_i11=INT_MIN;
    int dp_i20 = 0, dp_i21=INT_MIN;
    for(auto elem : prices) {
        dp_i10 = max(dp_i10,dp_i11+elem); 
        dp_i11 = max(dp_i11,-elem); 
        dp_i20 = max(dp_i20,dp_i21+elem); 
        dp_i21 = max(dp_i21,dp_i10-elem); 
    }
    return dp_i20;
}
```
## 14.[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

我们知道，买卖一次要两天，所以在给定的n个股票中，最多允许买卖n/2，如果给定的k超过n/2，那就是不限制买卖股票次数了，这样就可以回到第122题进行求解，否则套用123题的解法进行求解。

```cpp
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        if(n==0) return 0;
        int res=0;
        if(k > n/2) {
            int dp_0=0,dp_1=-prices[0];
            for(int i=1;i<n;i++) {
                dp_0 = max(dp_0,dp_1+prices[i]);
                dp_1 = max(dp_1,dp_0-prices[i]);
            }
            return dp_0;
        }
        vector<vector<vector<int>>> dp(n,vector<vector<int>>(k+1,vector<int>(2)));
        for(int i=0;i<n;i++) {
            for(int j=1;j<=k;j++) {
                if(i==0) {
                    dp[i][j][0] = 0;
                    dp[i][j][1] = -prices[0];
                    continue;
                }
                dp[i][j][0] = max(dp[i-1][j][0],dp[i-1][j][1]+prices[i]);
                dp[i][j][1] = max(dp[i-1][j][1],dp[i-1][j-1][0]-prices[i]);
            }
        }
        return dp[n-1][k][0];
    }
};
```
## 15.[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

> 动态规划

```cpp
class Solution {
public:
    // 动态规划
    // dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i])
    // dp[i][1] = max(dp[i-1][1],dp[i-1][2]-prices[i])
    // dp[i][2] = dp[i-1][0]
    int maxProfit(vector<int>& prices) {
        int n = prices.size();

        if(n==0) return 0; 

        vector<vector<int>> dp(n,vector<int>(3));
        dp[0][0] = 0;
        dp[0][2] = 0;
        dp[0][1] = -prices[0];
        for(int i=1;i<n;i++) {
            dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i]);
            dp[i][2] = dp[i-1][0];
            dp[i][1] = max(dp[i-1][1],dp[i-1][2]-prices[i]);
        }
        return dp[n-1][0];
    }
```
> 上述动态规划精简
```cpp
    // 上述动态规划精简
    // dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i])
    // dp[i][1] = max(dp[i-1][1],dp[i-2][0]-prices[i])
    int maxProfit(vector<int>& prices) {
        int n = prices.size();

        if(n==0) return 0; 

        vector<vector<int>> dp(n,vector<int>(3));
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        // 冷冻期设置
        int cold = 1;
        for(int i=1;i<n;i++) {
            dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i]);
            if(i-1<cold) {
                dp[i][1] = max(dp[i-1][1],dp[i-1][0]-prices[i]);
            } else {
                dp[i][1] = max(dp[i-1][1],dp[i-cold-1][0]-prices[i]);
            }
        }
        return dp[n-1][0];
    }
```
> 状态压缩
```cpp
    // 状态压缩
    int maxProfit(vector<int>& prices) {
        int n = prices.size();

        if(n==0) return 0; 

        int dp_i0=0;
        int dp_i1=-prices[0];
        int dp_i2=0;
        for(int i=0;i<n;i++) {
            int tmp = dp_i0;
            dp_i0 = max(dp_i0,dp_i1+prices[i]);
            dp_i1 = max(dp_i1,dp_i2-prices[i]);
            dp_i2 = tmp;
        }
        return dp_i0;
    }
};
```
