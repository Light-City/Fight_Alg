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

 