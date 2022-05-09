### 1. 反转链表

[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```c++
//迭代
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (head == NULL || head->next == NULL) return head;
        ListNode *slow = NULL;
        ListNode *fast = head;
        while(fast) {
            ListNode *temp = fast->next;
            fast->next = slow;
            slow = fast;
            fast = temp; 
        }
        return slow;
    }
};

//递归
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return head;
        ListNode *p = nullptr;
        ListNode *q = head;

        return re(p,q);
    }

    ListNode *re(ListNode *pre, ListNode *cur) {
        if(cur == nullptr) return pre;
        ListNode *tmp = cur->next;
        cur->next = pre;
        pre = cur;
        cur = tmp;
        return re(pre, cur);
    }
};
```



### 2. LRU缓存机制

[146. LRU 缓存](https://leetcode-cn.com/problems/lru-cache/)

```c++
class LRUCache {
public:
    int size;
    list<pair<int,int>> cache;  //key value
    unordered_map<int, list<pair<int,int>>::iterator> mapcache;  // key, cache::iterator

    LRUCache(int capacity) {
        size = capacity;
    }
  
    int get(int key) {
        auto it = mapcache.find(key);
        if (it == mapcache.end()) return -1;
        cache.splice(cache.begin(),cache,it->second); // o(1)
        return it->second->second;
    }
    
    void put(int key, int value) {
        auto it = mapcache.find(key);
        if(it != mapcache.end()) {
            it->second->second = value;
            cache.splice(cache.begin(),cache,it->second);
        }else {
            cache.insert(cache.begin(),make_pair(key,value));
            mapcache[key] = cache.begin();
            if(cache.size() > size) {
                mapcache.erase(cache.back().first); // o(1)
                cache.pop_back();
            }
        }

    }
};
```

### 3. 无重复字符的最长字串

[3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

还是比较难的

维护的cnt数组:cnt[i]表示如果出现字符i，那么起始下标就应该置为cnt[i]，即i上次出现的下一个位置

那么在遍历字符串s的过程中，我们对于每一个字符，更新 起始下标 与 cnt[i] 的位置；然后求最长长度即可

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int start = 0;
        int res =  0;
        vector<int> cnt(128,0);
        for(int i = 0; i<s.size(); i++) {
            start = max(start, cnt[s[i]]);  // 注意这里，要取两者的最大值
            cnt[s[i]] = i+1;
            res = max(res, i -start + 1);
        }
        return res;
    }
};
```



### 4. 数组中第k个最大元素

[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

快排or堆排序

```c++
//快排版
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        // 快排或者堆排序
        // 快排 -- 从大到小排序
        return quicksort(nums, 0 , nums.size()-1, k-1);
    }
    int quicksort(vector<int> &nums, int left, int right, int k) {
        int index = part(nums, left, right);
        if(index == k) {
            return nums[index];
        }else if (index > k) {
            return quicksort(nums, left, index-1, k);   // 要在这里返回
        }else {
            return quicksort(nums, index+1, right, k);
        }
        // return nums[index];  这样不行
    }

    int part(vector<int> &nums, int left, int right) {
        int x = nums[left];
        while(left < right) {
            while(left < right && nums[right] <= x) right--;
            nums[left] = nums[right];    // 要这样写
            while(left < right && nums[left] > x) left++;
            nums[right] = nums[left];
            // if(left < right) swap(nums[left], nums[right]);   这种不行
        }
        nums[left] = x;
        return left;
    }
};

//堆排序
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        // 快排or堆排
        // 堆排，可以直接使用大顶堆
        int len = nums.size();
        for(int i = len/2; i >=0 ;i--) {
            down(i, len, nums);
        }

        for(int i = 0;i < k-1;i++) {
            swap(nums[0], nums[len-i-1]);
            down(0, len - i - 1, nums);
        }
        return nums[0];
    }

    void down(int x, int len, vector<int> &nums) {
        int t = x;
        if(x*2+1 < len && nums[t] < nums[x*2+1]) t=x*2+1;
        if(x*2+2 < len && nums[t] < nums[x*2+2]) t=x*2+2;
        if(t != x) {
            swap(nums[t],nums[x]);
            down(t, len, nums);
        }
    }
};
```



### 5. K个一组翻转链表

[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group)

在做这题之前补了一下[92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/),因为反转链表II就是反转一组链表,不过指定了起始位置(left,right).

[反转链表II]

难点在于怎么使用o(1)的空间复杂度实现反转

注意这里与反转链表I不一样,这里是反转过后前后还有其他结点,所以在一次迭代的过程中,我们只能反转内部的结点

并且在反转完成后,我们要对本来前后的结点做处理,所以首先我们要保存left前的一个结点(first),right后的一个结点就不用保存(见下面一句话)

使用两个指针(l,r)反转,反转结束后,快指针r将指向right的后一个节点,反转后的第一个结点就是反转过程结束后的慢指针l指向的结点.

此时,我们就差一个结点未知,那就是反转后的最后一个结点(要指向right的后一个结点);注意,这个结点就是反转前的第一结点,所以我们把他另外再保存一份(second)即可.

```c++
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        if(left == right) return head;
        ListNode *dummy = new ListNode(0);
        dummy -> next = head;
        ListNode *first = dummy; 
        for(int i = 1;i < left;i++) {
            first = first->next;
        }
        ListNode *second = first->next;
        ListNode *l = second;
        ListNode *r = l->next;
        for(int i = left; i < right; i++) {
            ListNode *temp = r->next;
            r->next = l;
            l = r;
            r = temp;
        }
        first->next = l;
        second->next = r;
        return dummy -> next;
    }
};
```

[K 个一组翻转链表]

利用上面的反转链表II的思想,我们只要先统计出链表长度,计算出需要反转的次数,然后再在每次反转中按照上题进行反转即可

要稍微处理的就是,在每次反转结束后,处理完(left,right)两端的结点之后,要更新first和second的指针

每次反转要更新l,r指针即可.

递归法的思想就是传入left的前一个结点和right结点,然后处理好k个节点的反转,再递归

```c++
class Solution {
public:
    // 递归 or 迭代
    ListNode* reverseKGroup(ListNode* head, int k) {
        if(k == 1 || head -> next == nullptr) return head;
        ListNode *dummy = new ListNode(0);
        dummy->next = head;
        int cnt = 0;
        ListNode *temp = head;
        while(temp) {
            temp = temp -> next;
            cnt ++;
        }
        ListNode *first = dummy;
        ListNode *second = dummy->next;
        for(int i = 0; i < cnt / k; i++) {
            ListNode *l = second;
            ListNode *r= second->next;
            for(int j = 0; j < k-1 ;j++) {
                temp = r->next;
                r->next=l;
                l = r;
                r=temp;
            }
            first->next = l;
            second->next = r;
            first = second;
            second = r;
        }
        return dummy->next;
    }
};


//递归法
class Solution {
public:
    // 递归 or 迭代
    ListNode* reverseKGroup(ListNode* head, int k) {
        if(k == 1 || head -> next == nullptr) return head;

        ListNode *dummy = new ListNode(0);
        dummy->next = head;

        ListNode *r = dummy;
        for(int i = 0; i < k;i++) {
            r = r->next;
        }
        reverseK(dummy, r,k);
        return dummy->next;
    }
    void reverseK(ListNode *preleft, ListNode *right,int k) {
        if(right == nullptr || preleft == nullptr) return;
        ListNode *first = preleft;
        ListNode *second = preleft->next;
        ListNode *l = second;
        ListNode *r = second->next;

        ListNode *rightNext = right->next;
        while(r != rightNext) {
            ListNode *temp = r->next;
            r->next = l;
            l = r;
            r = temp;
        }
        first->next = l;
        second->next = r;
        for(int i = 1; i < k; i++) {
            if(r != nullptr) r = r->next;
            if(r == nullptr) break;
        }
        reverseK(second, r, k);
    }
};
```



### 6. 三数之和

[15. 三数之和](https://leetcode-cn.com/problems/3sum/)

主要思想：

	- 首先排序，用于去重
	- 对于第一个元素的选取要去重
	- 然后利用双指针从两端开始查找数据，注意到是两个指针遍历过程中也要去重

```c++
class Solution {
public:
    //双指针更简单
    //这里主要是去重操作
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        int len = nums.size();
        if(len < 3 || nums[0] > 0 || nums[len-1] < 0) return res;
        for(int i = 0;i < len;i ++) {
            if(nums[i] > 0) break;
            if(i != 0 && nums[i-1] == nums[i]) continue;
            int target = 0 - nums[i];
            int left = i + 1;
            int right = len - 1;
            while(left < right) {
                if ((left > i+1 && nums[left] == nums[left - 1]) || nums[left] + nums[right] < target) left++;
                else if ((right < len - 1 && nums[right] == nums[right+1]) || nums[left] + nums[right] > target) right--;
                else {
                    vector<int> v;
                    v.push_back(nums[i]);
                    v.push_back(nums[left]);
                    v.push_back(nums[right]);
                    res.push_back(v);
                    left++;
                }
            }
       }
       return res;
    }
};
```

### 7. 手撕快排

[912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

```c++
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        quick_sort(nums,0,nums.size()-1);
        return nums;
    }
    void quick_sort(vector<int> &nums, int l, int r) {
        if(l >= r) return;
        int mid = (l+r) / 2;
        int x = nums[mid];
        int i = l-1;
        int j = r+1;
        while(i < j) {
            while(nums[++i] < x);
            while(nums[--j] > x);
            if(i < j) swap(nums[i], nums[j]); 
        }
        quick_sort(nums, l, j);
        quick_sort(nums, j+1, r);
    }
};
```

### 8. 最大子数组和

[53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

累加小于0则重新计算即可；过程中保存最大值。

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        int res = INT_MIN;
        int cnt = 0;
        for(int i = 0; i < len; i++ ){
            cnt += nums[i];
            res = max(res, cnt);
            if(cnt <= 0) {
                cnt = nums[i] > 0 ? nums[i] : 0;
            }
        }
        return res;
    }
};
```

### 9. 合并两个有序链表

[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

用一个结点穿插于两个链表之间，用双指针遍历两个链表

```c++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode *p1 = list1;  //
        ListNode *p2 = list2;
        ListNode *r = new ListNode(0);
        ListNode *res = r;
        while(p1 && p2) {
            if(p1->val < p2->val) {
                r->next = p1;
                r = r->next;
                p1 = p1->next;

            }else {
                r->next = p2;
                r = r->next;
                p2 = p2->next;
            }
        }
        if(p1) {
            r->next = p1;
        }
        if(p2) {
            r->next = p2;
        }
        return res->next;
    }
};
```

### 10. 两数之和

[1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

用hashmap记录已经遍历过的结点，同时查找另一个数

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> re;
        for(int i = 0; i< nums.size();i++) {
            int n = target - nums[i];
            if(re.find(n) == re.end()){
                re.insert(pair(nums[i], i));
            }else {
                return {re[n],i};
            }
        }
        return {-1,-1};
    }
};
```

### 11. 二叉树层次遍历

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> qe;
        vector<vector<int>> res;
        if(root == nullptr) return res;
        qe.push(root);
        res.push_back({root->val});

        while(!qe.empty()){
            int len =qe.size();
            vector<int> temp;
            while(len--){
                TreeNode *node= qe.front();
                qe.pop();
                if(node->left) {
                    qe.push(node->left);
                    temp.push_back(node->left->val);
                }
                if(node->right) {
                    qe.push(node->right);
                    temp.push_back(node->right->val);
                }
            }
            if(temp.size() != 0) res.push_back(temp);
        }
        return res;
    }
};
```

### 12. 环形链表

[141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

注意的就是遍历过程中快慢指针都不为null

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *dummy = new ListNode(0);
        dummy->next = head;
        ListNode *p = head;
        while(p && dummy) {
            if(p == dummy) {
                return true;
            }
            if(p->next)
                p = p->next->next;
            else return false;
            dummy = dummy->next;
        }
        return false;
    }
};
```

### 13. 买卖股票的最佳时机

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock)

不限制买卖的次数，所以每次当后面一天的价格高于前面一天的就进行利润累加即可

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() == 1) return 0;
        int buy = prices[0];
        int profit = 0;
        for(int i = 1; i< prices.size(); i++){
            if(prices[i] < buy){
                buy = prices[i];
            }else {
                profit = max(profit, prices[i] - buy);
            }
        }
        return profit;
    } 
};
```

### 14. 相交链表

[160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        int lenA = getlength(headA);
        int lenB = getlength(headB);

        ListNode *pa = headA;
        ListNode *pb = headB;

        while(lenA != lenB && pa && pb) {
            if(lenA < lenB) {
                pb = pb->next;
                lenB --;
            }
            else {
                pa = pa->next;
                lenA --;
            }
        }
        while(pa != pb && pa && pb) {
            pa = pa->next;
            pb = pb->next;
        }

        if(pa == nullptr || pb == nullptr) return nullptr;
        return pa;
    }
    int getlength(ListNode *node) {
        ListNode *temp = node;
        int n = 0;
        while(temp) {
            temp = temp->next;
            n++;
        }
        return n;
    }

};
```

### 15. 二叉树的锯齿形层序遍历

[二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

奇数偶数的从不同方向进行遍历（双向队列）

要注意的是，不同方便遍历的时候，入栈的方向也不同

```c++
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        deque<TreeNode *> dq;
        vector<vector<int>> res;
        if(root == nullptr) return res;
        dq.push_back(root);
        int cnt = 0;
        while(!dq.empty()) {
            int len = dq.size();
            vector<int> vt;
            while(len --) {
                TreeNode *temp = nullptr;
                if(cnt % 2 == 0) {
                    temp = dq.front();
                    dq.pop_front();
                    if(temp->left) {
                        dq.push_back(temp->left);
                    }
                    if(temp -> right) {
                        dq.push_back(temp->right);
                    }
                }else {
                    temp = dq.back();
                    dq.pop_back();
                    if(temp -> right) {
                        dq.push_front(temp->right);
                    }
                    if(temp->left) {
                        dq.push_front(temp->left);
                    }
                }
                vt.push_back(temp->val);
            }
            cnt++;
            res.push_back(vt);
        }
        return res;
    }
};
```

### 16. 合并两个有序数组

[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

注意指针的处理

```c++
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int len1 = m - 1;
        int len2 = n - 1;
        int cnt = m + n -1;
        while(len1 >=0 && len2 >= 0) {
            nums1[cnt--] = nums1[len1] < nums2[len2] ? nums2[len2--] : nums1[len1--];
        }
        while(len2>=0){
            nums1[cnt--] = nums2[len2--];
        }

    }
};
```

### 17. 二叉树的最近公共祖先

[236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == NULL || root == p || root == q) return root;
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);

        if(left && right) return root;
        else if(left && right == NULL) return left;
        else if(left ==  NULL && right) return right;
        else return NULL;

    }

};
```

### 18. 有效的括号

[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```c++
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        int len = s.size();
        if (len % 2 == 1) return false;
        for(int i = 0;i < len; i++){
           if(s[i] == '(' || s[i] == '{' ||s[i] == '[' ){
               st.push(s[i]);
           }
           if(st.size() == 0) return false;
           if(s[i] == ')' ){
               char t = st.top();
            //    cout << t << endl;
               if(t != '(') return false;
               st.pop();
           }
           if(s[i] == '}' ){
               char t = st.top();
               if(t != '{') return false;
               st.pop();
           }
           if(s[i] == ']' ){
               char t = st.top();
               if(t != '[') return false;
               st.pop();
           }

        }
        return st.empty();
    }
};
```

### 19. 最长回文串

[5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

这里要注意的是判断回文子串的方式，通过当前`s[i]`与`s[j]`是否相等以及 `s[i+1]-s[j-1]`之间的字符串是否是回文来判断

这就要求当遍历`dp[i][j]`的时候`dp[i+1][j-1]`以及被处理过，也就是i从大到小，j从小到大

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        //dp法
        int len = s.size();
        if(len == 0 || len == 1) return s;
        int start=0, end=0;
        int max = INT_MIN;
        string res = string(1,s[0]);
        vector<vector<int>> dp(len, vector<int>(len, 0));

        for(int i = len - 1; i >= 0; i--) {
            for(int j = i; j < len; j++) {
                if(i == j) {
                    dp[i][j] = 1;
                }else {
                    if(s[i]==s[j] && (j - i < 2 || dp[i+1][j-1])){
                        dp[i][j] = 1;
                        if(j - i > max) {
                            max = j - i;
                            start = i;
                            end = j;
                        }
                    }
                }
            }
        }
        return s.substr(start, end - start + 1);  //param->(start,len)
    }
};
```

### 20. 搜索旋转排序数组

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

主要是二分的思想，主要是先判断出左边有序还是右边有序，然后在有序的部分进行判断，否则就进入另一部分

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int r = nums.size() - 1;
        int l = 0;
        while(l <= r) {
            int mid  = (l+r) / 2;
            if(nums[mid] == target) return mid;
            // 要先在一个有序的部分里去判断，不然就无法判断出应该去左边还是右边查找
            if(nums[mid] < nums[r]) {
                if(target > nums[mid]  && target <= nums[r]) {   // 这里有个等号
                    l = mid + 1;
                }else {
                    r = mid - 1;
                }
            }else {
                if(target < nums[mid] && target >= nums[l]) { // 这里有个等号
                    r = mid - 1;
                }else {
                    l = mid + 1;
                }
            }
        }
        return -1;
    }
};
```



### 21. 岛屿数量

[200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

其实就是求不连通的1块的数量，

bfs,然后每次把1置为0。把全部1置为0需要的遍历次数即为所求

```c++
class Solution {
public:
    // vector<vector<int>> used(301, vector<int>(301,0));
    int m,n;
    int numIslands(vector<vector<char>>& grid) {
        m = grid.size();
        n = grid[0].size();
        int res  = 0;
        for(int i = 0;i < m; i++) {
            for(int j = 0;j < n; j++) {
                if(grid[i][j] == '1') {
                    res ++;
                    traverse(grid,i,j);
                }
            }
        }
        return res;
    }
    void traverse(vector<vector<char>> &grid, int x, int y) {
        if(x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == '0') {
            return;
        }
        else {
            grid[x][y] = '0';
            traverse(grid, x, y-1);
            traverse(grid, x-1, y);
            traverse(grid, x, y+1);
            traverse(grid, x+1, y);
        }
    }
};
```

### 22. 全排列

[46. 全排列](https://leetcode-cn.com/problems/permutations/)

这里数据不重复，所以就不用去重

然后遍历的过程中，虽然当前元素不用再去，但是有可能取完i+2再取i+1

所以还是要每次回溯都从下标0开始，然后对以及访问过数组使用标记即可

```c++
class Solution {
    public:
        vector<int> temp;
        vector<vector<int>> res;
        int used[21];

        vector<vector<int>> permute(vector<int>& nums) {
            int len = nums.size();
            if(len == 0) return res;
            backtracking(nums);
            return res;        
        }


        void backtracking(vector<int> &nums) {
            if(temp.size() ==  nums.size()) {
                res.push_back(temp);
                return;
            }else {
                for(int i = 0; i < nums.size(); i++) {
                    if(used[nums[i] + 10] == 0) {
                        used[nums[i] + 10] = 1;
                        temp.push_back(nums[i]);
                        backtracking(nums);
                        temp.pop_back();
                        used[nums[i] + 10] = 0;
                    }else {
                        continue;
                    }
                }
            }
        }
    };
```

 

### 23. 字符串相加

[415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

基本思想就是每一位进行相加，最后再拼接字符串

```c++
class Solution {
public:
    string addStrings(string num1, string num2) {
        int cnt = 0;
        string res = "";

        int len1 = num1.size() - 1;
        int len2 = num2.size() - 1;

        while(len1>=0 || len2>=0) {
            char a = '0', b = '0';
            if(len1 >= 0)  a = num1[len1--];
            if(len2 >= 0)  b = num2[len2--];
            int t = a - '0' + (b - '0');
            t = t + cnt;
            if(t > 9) {
                cnt = 1;
                t = t - 10;
            }else {
                cnt = 0;
            }
            string ts = string(1, t + '0');
            res += ts;
        }
        if(cnt == 1) res += "1";
        reverse(res.begin(), res.end());
        return res;
    }
};
```

### 24. 合并k个升序链表

[23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

先把k个链表的第一个元素取出来，进行堆排序；然后每次弹出第一个元素之后，把该元素所在链表的下一个元素放入堆中，进行调正

一直到所有元素弹出完成（每次弹出时建立新的链表）

```c++
class Solution {
public:
    ListNode* h[10010];
    int k;
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode *dummy = new ListNode(0);
        k = lists.size();
        if(k == 0) return nullptr;
        int cnt = 1;
        for(int i = 0; i < lists.size(); i++) {
            if(lists[i]){
                h[cnt++] = lists[i];
            }
            else {
                k--;
            }
        }
        for(int i = k/2; i >= 1; i--) {
            down(i);
        }
        ListNode *head = dummy;
        while(k) {
            ListNode *temp = h[1];
            head->next = temp;
            head = temp;
            if(temp->next) {
                h[1] = temp->next;
                down(1);
            }else {
                h[1] = h[k--];
                down(1);
            }
        }

        return dummy->next;
    }
    void down(int x) {
        int t = x;
        if(2 * x <= k && h[t]->val > h[2*x]->val) t = 2*x;
        if(2 * x + 1 <= k && h[t]->val > h[2*x+1]->val) t = 2*x+1;
        if(x != t) {
            swap(h[x],h[t]);
            down(t);
        }
    }

};
```

### 25. 反转链表II

[92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

首要要记录反转起始结点的前一个结点，反转的起始结点（保存2份，一份用于遍历过程中的反转），反转的起始节点的下一个结点

```c++
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        if(left == right) return head;
        ListNode *dummy = new ListNode(0);
        dummy->next = head;

        ListNode *first = dummy;
        for(int i = 1; i < left; i++) {
            first = first->next;
        }
        
        ListNode *second = first->next;
        ListNode *l = second;
        ListNode *r = l->next;

        for(int i = left; i < right; i++) {
            ListNode *temp = r->next;
            r->next = l;
            l = r; //不能用l = l->next，l->next可能已经反转了
            r = temp;
        }

        first->next = l;
        second->next = r;

        return dummy->next;
    }
};
```



### 26. 环形链表II

[142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if (head == NULL) return NULL;
        ListNode *slow = head;
        ListNode *fast = head;
        
        while(fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if(slow == fast) break;
        }
        if(fast == NULL || fast->next == NULL) return NULL;
        slow = head; 
        while(slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        return fast;
    }
};
```

### 27. 螺旋矩阵

[54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

设立四个边界即可

```c++
class Solution {
public:
    vector<int> res;

    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        int u = 0;
        int b = m - 1;
        int l = 0;
        int r = n - 1;
        while(1) {
            for(int i = l; i <= r;i++) {
                res.push_back(matrix[u][i]);
            }
            // u++;
            if(++u > b) break;
            for(int i = u; i <= b;i ++) {
                res.push_back(matrix[i][r]);
            }
            if(--r < l) break;
            for(int i =r; i >= l; i--) {
                res.push_back(matrix[b][i]);
            }
            if(--b < u) break;
            for(int i =b; i >= u;i--) {
                res.push_back(matrix[i][l]);
            }
            if(++l > r) break;
        }
        return res;
    }
};
```

### 28. 最长上升子序列

[300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

dp[i]表示从nums[0]开始以nums[i]结尾的元素的最长上升子序列的长度	

```c++
class Solution {
public:
    int dp[2501];
    int lengthOfLIS(vector<int>& nums) {
        int len = nums.size();
        if(len == 1) return 1;
        int res = INT_MIN;
        for(int i = 0; i < len; i++) {
            dp[i] = 1;
            for(int j = 0; j <= i; j++){
                if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j]+1);
            }
            res = max(res, dp[i]);
        }
        return res;
    }
};
```

### 29. 接雨水

[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

最优法是双指针法，也可以使用dp法

【dp法】

进行两轮循环，每轮循环记录左边或者右边比当前元素大的元素

第三轮循环进行最终求解，每个记录能累计的雨水为min(当前元素左边的最大值，当前元素右边的最大值)-当前元素的高度

时间O(n),空间O(n)

【双指针法】



```c++
class Solution {
public:
    int trap(vector<int>& height) {
        // dp 不那么典型
        int len = height.size(); 
        // if (height == NULL) return 0;
        int res = 0;
        vector<int> leftnums(len), rightnums(len);
        leftnums[0] = height[0];
        for(int i = 1; i< len;i++){
            leftnums[i] = max(leftnums[i-1], height[i]);
        }
        rightnums[len-1] = height[len-1];
        for(int i = len - 2; i >= 0;i --) {
            rightnums[i] = max(rightnums[i+1], height[i]);
        }
        for(int i = 0; i < len;i ++) {
            res += min(leftnums[i],rightnums[i])- height[i];
        }
        return res;
    }
};	

//双指针
class Solution {
public:
    int trap(vector<int>& height) {
        // 双指针
        int left = 0;
        int right = height.size() - 1;
        int maxleft = 0;
        int maxright = 0;
        int res = 0;
        while(left < right) {
            if(height[left] < height[right]) {
                if(maxleft < height[left]) {
                    maxleft = height[left];
                }else {
                    res += (maxleft - height[left]);
                }
                left ++;
            }else {
                if(maxright < height[right]) {
                    maxright = height[right];
                }else {
                    res += (maxright - height[right]);
                }
                right--;
            }
        }
        return res;
    }
};
```

### 30. 二分查找

[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

```c++
func search(nums []int, target int) int {
    left := 0
    right := len(nums) - 1

    for left <= right {
        mid := (left + right) /2
        if nums[mid] == target {
            return mid
        }else if nums[mid] < target {
            left = mid + 1
        }else {
            right = mid - 1
        }
    }
    return -1
}
```

### 31. 二叉树的中序遍历

主要是非递归写法

```c++
class Solution {
public:
    vector<int> res;
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode *> st;
        TreeNode *first = root;
        while(!st.empty() || first) {
            while(first != nullptr ) {
                st.push(first);
                first = first -> left;
            }   
            TreeNode *node = st.top();
            st.pop();
            res.push_back(node->val);
            first = node->right;
        }
        return res;
    }
};
```

### 32. 用栈实现队列

[232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

用2个栈实现

```c++
class MyQueue {
public:
    stack<int> st1;
    stack<int> st2;

    MyQueue() {

    }
    void push(int x) {
        st1.push(x);
    }
    int pop() {
        if(!st2.empty()) {
            int x = st2.top();
            st2.pop();
            return x;
        }else {
            while(!st1.empty()) {
                int x = st1.top();
                st2.push(x);
                st1.pop();
            }
            int x = st2.top();
            st2.pop();
            return x;
        }
    }
    int peek() {
        if(!st2.empty()) {
            int x = st2.top();
            return x;
        }else {
            while(!st1.empty()) {
                int x = st1.top();
                st2.push(x);
                st1.pop();
            }
            int x = st2.top();
            return x;
        }
    }
    bool empty() {
        return st1.empty() && st2.empty();
    }
};
```

### 33. 重排链表

[143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

主要的思路是先把后半部分的链表反转，然后和前半部分的链表进行交叉连接即可

```c++
class Solution {
public:
    void reorderList(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return;
        ListNode *fast = head->next;
        ListNode *slow=  head;
        while(fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        ListNode *reverseNode = slow->next;
        slow->next = nullptr;  //注意这里的处理
        ListNode *head2 = Revs(reverseNode);

        ListNode *p1 = head;
        ListNode *p2 = head2;
        while(p2 != nullptr) {
            ListNode *temp = p1->next;
            ListNode *temp2 = p2->next;
            p1->next = p2;
            p2->next = temp;
            p1 = p2->next;
            p2 =  temp2;
        }

    }

    ListNode* Revs(ListNode *node) {
        ListNode *slow = nullptr;
        ListNode *fast = node; 
        while(fast) {
            ListNode *temp = fast->next;
            fast->next = slow;
            slow = fast;
            fast = temp;
        }
        return slow;
    }
};
```



### 34. 二叉树的右视图

[199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

层次遍历即可

```c++
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        queue<TreeNode *> qu;
        vector<int> res;
        if(root == nullptr) return res;

        qu.push(root);
        int flag = 0;
        while(!qu.empty()){
            int len = qu.size();
            while(len --) {
                TreeNode *temp = qu.front();
                qu.pop();
                if(len == 0) {
                    res.push_back(temp->val);
                }
                if(temp->left) {
                    qu.push(temp->left);
                }
                if(temp->right) {
                    qu.push(temp->right);
                }
            }
        }
        return res;
    }
};
```

### 35. 二叉树的最大路径和

[124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

```c++
class Solution {
public:
    int res = INT_MIN;
    int maxPathSum(TreeNode* root) {
        if(root == nullptr) return 0;
        dfs(root);
        return res;
    }

    int dfs(TreeNode *root) {
        if(root == nullptr) return 0;
        int left = max(0, dfs(root->left));
        int right = max(0, dfs(root->right));
        res = max(res, root->val + left + right);
        return max(left,right) + root->val;
    }
};
```

### 36. 爬楼梯

[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

```c++
class Solution {
public:
    int climbStairs(int n) {
        int dp[46];
        dp[1] = 1;
        dp[2] = 2;
        for(int i = 3; i <= n;i++){
            dp[i] = dp[i-1] + dp[i-2]; 
        }
        return dp[n];
    }
};
```

### 37. 合并区间

[56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

```c++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if(intervals.empty())
            return {};
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> res;
        res.push_back(intervals[0]);
        for(int i = 1; i < intervals.size(); i++) {
            if(res.back()[1] >= intervals[i][0]) {
                res.back()[1] = max(res.back()[1], intervals[i][1]);
            }else {
                res.push_back(intervals[i]);
            }
        }
        return res;
    }
};
```

### 38. 链表中倒数第k个节点

[剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

```c++
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        if (head == NULL) return head;

        ListNode *fast = head;
        for(int i = 0;i < k;i ++) {
            fast = fast -> next;
        }
        // cout << fast->val;

        while(fast) {
            head = head->next;
            fast = fast->next;
        }
        return head;
    }
};
```

### 39. 删除链表的倒数第 N 个结点

[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        int cnt = n;
        // ListNode *p = head;
        if(head->next == nullptr) return nullptr;
        ListNode *dummy = new ListNode(0);
        dummy->next = head;
        ListNode *p = dummy;
        for(int i = 0; i < cnt;i++) {
            p = p->next;
        }
        // cout << p ->val << endl;
        ListNode *slow = dummy;
        while(p->next) {
            p = p->next;
            slow = slow->next;
        }
        // cout << slow -> val << endl;
        slow->next = slow->next->next;
        return dummy->next;

    }
};
```

### 40. x 的平方根 

[69. x 的平方根 ](https://leetcode-cn.com/problems/sqrtx/)

能优化的地方就在于可以使用二分

即当中间值的平方大于或者小于x时，可以对应的去左边或者右边查找

要注意的mid的求解，这里可能会出现left，right之间只有2个值死循环的问题，所以mid要上取整

```c++
class Solution {
public:
    int mySqrt(int x) {
        if(x == 0) return 0;
        if(x == 1) return 1;
        int res = 0;
        int left = 1;
        int right = x/2;
        while(left < right) {
            int mid = (left+right) / 2 + 1; //向上取整，因为当left和right之间一共就2个数的时候，下面的划分会导致死循环
            if(mid == x / mid) return mid;
            else if(mid > x / mid) {
                right = mid-1;
            }else {
                left = mid;
            }
        }
        return left;
    }
};
```

### 41. 字符串转换整数 (atoi)

[8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

主要是思路要理顺

1. 去掉前导空格，记录新的起始idx
2. 读取正负号；如果不是正负号也不是数字就说明是无效的，返回0
3. 读取每个数字进行转换，要注意的就是溢出的处理（不是数字的时候就停止转换，直接截断）

```c++
class Solution {
public:
    int myAtoi(string s) {
        int n = s.size();
        int idx = 0;
        while(idx < n && s[idx] == ' ') {
            idx ++; //去除前导空格
        }
        if(idx == n) {
            return 0;
        }
        bool neg = 0;
        if(s[idx] == '-') {
            neg = 1;
            idx++;
        }else if(s[idx] == '+') {
            idx ++;
        }else if(s[idx] > '9' || s[idx] < '0') {
            return 0;
        }
        int ans = 0;
        while(idx < n && (s[idx] <= '9' && s[idx] >= '0')){
            int digit = s[idx] - '0';
            if(ans > (INT_MAX - digit) / 10){
                return neg ? INT_MIN : INT_MAX;
            }
            ans  = ans * 10 + digit;
            idx++;
        }
        return neg?-ans:ans;
    }
};
```

### 42. 删除排序链表中的重复元素 II

[82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

【迭代法】

slow指向最后一个不重复的结点，fast用于遍历重复结点，flag用于表明当前不重复的情况下是否以及删除了重复节点

【递归法】

每次返回不重复的结点，如果重复就遍历到不重复的结点，然后进行递归删除返回

```c++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode *dummy = new ListNode(-101);
        dummy->next=head;
        int flag = 0;
        ListNode *slow = dummy;
        ListNode *fast = head;
        while(slow && fast && fast->next) {
            if(fast->val == fast->next->val) {
                fast->next = fast->next->next;
                if(fast->next == nullptr) {
                    slow->next = nullptr;
                }
                flag = 1;
            }
            else if (flag == 1) {
                slow->next = fast->next;
                fast = fast->next;
                flag = 0;
            }else {
                slow = slow->next;
                fast = fast->next;
                flag = 0;
            }
        }
        return dummy->next;
    }
};

//递归法
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(!head) {
            return head;
        }
        if(head->next && head->val == head->next->val) {
            while(head->next && head->val == head->next->val) {
                head = head->next;
            }
            return deleteDuplicates(head->next);
        }else {
            head->next = deleteDuplicates(head->next);
        }
        return head;
    }  
};
```



### 43. 排序链表

[148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

使用并归，从步长为1开始，每次x2，开始并归排序

这里需要两个函数---用于划分链表的cut()和合并两个有序链表的merge()

对于每一个步长，都要对整个链表所划分的所有段进行合并

在具体的实现中还要注意：

 - 每次的步长开始时，都是从头节点开始
 - 每次步长的遍历中，要有当前的这次两两合并的起始节点，以及上一次两两合并之后的最后一个结点（因为要把前面合并的结果和现在的结果进行连接）

```c++
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        ListNode *dummy = new ListNode(0);
        dummy->next = head;
        auto p = head;
        int length = 0;
        while(p) {
            length++;
            p = p->next;
        }
        for(int i = 1;i < length; i = i * 2) {
            auto cur = dummy->next;
            auto tail = dummy;
            while(cur) {
                auto left = cur;
                auto right = cut(left, i);
                cur = cut(right, i);

                tail->next = merge(left,right);
                while(tail->next) {
                    tail = tail->next;
                }
            }
        }
        return dummy->next;
    }
    ListNode* cut(ListNode *head, int step) {
        ListNode *pre = new ListNode(0);
        pre->next = head;
        ListNode *cur = head;
        while(pre && cur && step--) {
            pre = pre->next;
            cur = cur->next;
        }
        pre->next = nullptr;
        return cur;
    }
    // 有序链表的合并  返回合并之后的头结点
    ListNode* merge(ListNode* p1, ListNode* p2) {
        ListNode *temp = new ListNode(0);
        ListNode *dummy = temp;

        while(p1 && p2) {
            if(p1->val < p2->val) {
                temp->next = p1;
                p1 = p1->next;

            }else  {
                temp->next = p2;
                p2 = p2->next;
            }
            temp = temp->next;
        }
        if(p1) {
            temp->next = p1;
        }
        if(p2) {
            temp->next = p2;
        }
        return dummy->next;
    }
};



// 递归写法
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if(head == nullptr) return head;
        return mergesort(head);
    }

    ListNode* mergesort(ListNode *head) {
        if(head->next == nullptr) return head;   // 注意这里是 head->next == null

        ListNode *p = head;
        ListNode *q = head;
        ListNode *pre = nullptr;
        while(p && q && q->next) {
            pre = p; 
            p = p->next;
            q = q->next->next;
        }
        if(pre != nullptr)
            pre->next = nullptr; // 分段 --- 最后相当于全部拆成单个节点，然后合并

        ListNode *l = mergesort(head);  //这里要记录返回值，因为合并之后的头节点不确定，合并之后的连接关系也只能通过返回的结点找到
        ListNode *r = mergesort(p);

        // merge()
        return merge(l, r);
    }

    //有序链表合并
    ListNode* merge(ListNode *l, ListNode* r) {
        ListNode *dummy = new ListNode(0);
        ListNode *cur = dummy;
        while(l && r) {
            if(l->val <= r->val) {
                cur->next = l;
                cur = l;
                l = l->next;
            }else {
                cur->next = r;
                cur = r;
                r = r->next;
            }
        }
        if(l != nullptr) {
            cur->next = l;
        }
        if(r != nullptr) {
            cur->next = r;
        }
        return dummy->next;
    }
};
```



### 44. 编辑距离

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

这里要注意的是`dp[i][j]`表示的是word1的前i的字符和word2的前j个字符使其相等需要的最少操作数

所以i=1,j=1时表示的是word1中的下标0和word2中的下标0对应的字符

而我们初始化需要初始化i=0和j=0的情况，即word1为空或者word2为空的情况

```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int len1 = word1.size();
        int len2 = word2.size();
        int len = max(len1, len2);
        vector<vector<int>> dp(len+1, vector<int>(len+1,0));
        
        for(int i = 0;i <= len;i++) {
            dp[i][0] = i;
            dp[0][i] = i;
        }

        for(int i = 1;i <= len1;i ++) {
            for(int j = 1; j <= len2;j++) {
                if(word1[i-1] == word2[j-1]) dp[i][j] = dp[i-1][j-1];
                else {
                    dp[i][j]=min(dp[i-1][j-1]+1, min(dp[i-1][j],dp[i][j-1]) + 1);
                }     
            }
        }
        return dp[len1][len2];
    }
};
```

### 45. 寻找两个正序数组的中位数

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

想象成一个二分问题，每次找第k个数，从nums1的下标为0的数 和 nums2的下标为0的数开始



- 如果 nums1当前下标后面的第k/2个数mid1 小于 nums2当前下标后面的第k/2个数mid2， 说明第k个数，一定不在num1当前位置后的k/2个数内，从而缩小区间

- 如果 nums1当前下标后面的第k/2个数mid1 大于等于 nums2当前下标后面的第k/2个数mid2， 说明第k个数，一定不在num2当前位置后的k/2个数内，从而缩小区间



特殊情况的处理：

- k==1，即nums1当前位置的第一个元素（其本身） 与 nums2当前位置的第一个元素（其本身）中选择较小值返回

- 当i，j已经超过了其数组的长度，那么就在另一个数组中直接取即可

    

- 当nums1或者nums2后的第k/2个元素已经超过了其数组的长度，那么就只能排除另一个数组里的k/2个数据

    - 以为要找第k个，nums1后面不足k/2个，所以
        - 如果nums2中的k/2个都在nums1剩余数字的左侧，那么第k个一定不再这k/2个里面
        - 如果nums2中的k/2个都在nums1剩余数字的右侧，那么加起来也小于k，第k个也一定不在nums2后面的k/2个里面


```c++
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int len1 = nums1.size();
        int len2 = nums2.size();
        int len = len1+len2;
        
        return (findk(nums1,0,nums2,0,(len+1)/2)  + findk(nums1,0,nums2,0,(len+2)/2) ) /2.0;   // 2.0 不是2
    }
    int findk(vector<int> &nums1, int i, vector<int> &nums2, int j, int k) {
        if(i >= nums1.size()) return nums2[j + k - 1];
        if(j >= nums2.size()) return nums1[i + k - 1];
        if(k == 1) {
            return min(nums1[i], nums2[j]);
        }
        int mid1 = (i+k/2-1) >= nums1.size() ? INT_MAX : nums1[i+k/2-1];
        int mid2 = (j+k/2-1) >= nums2.size() ? INT_MAX : nums2[j+k/2-1];
        if(mid1 < mid2) {
            return findk(nums1, i+k/2, nums2, j, k - k/2);
        }else {
            return findk(nums1, i, nums2, j+k/2, k - k/2);
        }
    }
};
```



### 46. 两数相加

[2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

```c++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *head1 = l1;
        ListNode *head2 = l2;
        int cnt = 0;
        while(l1 && l2) {
            int x = l1->val + l2->val + cnt;
            if(x > 9) {
                x = x%10;
                cnt = 1;
            }else {
                cnt = 0;
            }
            l1->val = x;
            l2->val = x;
            l1 = l1->next;
            l2 = l2->next;
        }
        if(!l1 && !l2 && cnt == 1) {
            ListNode *tt = head1;
            while(head1->next) {
                head1= head1->next;
            }
            head1->next = new ListNode(1);
            return tt;
        }

        if(l1) {
            ListNode *t = new ListNode(0);
            while(l1) {
                int x = l1->val + cnt;
                if(x > 9) {
                    cnt = 1;
                    x = 0;
                }else {
                    cnt = 0;
                }
                l1->val = x;
                if(l1->next == nullptr) {
                    t = l1;
                }
                l1 = l1->next;
            }
            if(cnt == 1) {
                ListNode *t1 = new ListNode(1);
                t->next = t1;
            }
            return head1;
        }
        else {
            ListNode *t = new ListNode(0);
            while(l2) {
                int x = l2->val + cnt;
                if(x > 9) {
                    cnt = 1;
                    x = 0;
                }else {
                    cnt = 0;
                }
                l2->val = x;
                
                if(l2->next == nullptr) {
                    t = l2;
                }
                l2 = l2->next;
            }
            if(cnt == 1) {
                ListNode *t1 = new ListNode(1);
                t->next = t1;
            }
            return head2;
        }
    }
};
```

### 47. 字符串转换整数 (atoi)

[8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

```c++
class Solution {
public:
    int myAtoi(string s) {
        int index = 0;
        while(index < s.size() && s[index] == ' '){
            index++;
        }
        int flag = 1;
        if(s[index] == '-') {
            flag = -1;
            index++;
        }else if(s[index] == '+') {
            index++;
        }else if(s[index] > '9' || s[index] < '0') return 0;
        int res = 0;
        while(index < s.size() && isdigital(s[index])) {
            int d = s[index] - '0';
            if(res > (INT_MAX - d)/10) {
                return flag == -1 ? INT_MIN : INT_MAX;
            }
            res = res * 10 + d;
            index++;
        }
        return flag * res;
    }
    bool isdigital(char c) {
        if(c >= '0' && c <= '9') return true;
        return false;
    }
};
```

### 48. 最长公共子序列

[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

dp问题，主要是初始化

这里直接使用全局变量，自动初始化为0

然后`dp[0][j]`和`dp[i][0]`的情况和后面其他的情况一致，所以可能直接使用`dp[i+1][j+1]`方式进行求解

```c++
class Solution {
public:
    int dp[1010][1010];
    int longestCommonSubsequence(string text1, string text2) {
        int len1 = text1.size();
        int len2 = text2.size();
        for(int i = 0;i < len1;i++) {
            for(int j = 0;j < len2;j++) {
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1]);
                if(text1[i] == text2[j]) {
                    dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j]+1);
                }
            }
        }
        return dp[len1][len2];
    }
};
```

### 49. 下一个排列

[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

主要是 规律是什么？

从后面向前遍历，然后找到第一个nums[i] 比nums[i-1]大的值，然后把nums[i-1]和后面从后往前第一个比他大的值进行交换

最后把i-length之间数排序即可

```go
func nextPermutation(nums []int)  {
    var reverse func(l,r int)
    reverse = func(l, r int) {
        for i:=l; i <= (r+l)/2;i++ {
            nums[i], nums[r+l-i] = nums[r+l-i], nums[i]
        }
    }
    flag := 0
    length := len(nums)
    for i:=length-1; i >= 1; i-- {
        if(nums[i] > nums[i-1]) {
            flag = 1;
            index := length-1
            for ;index>=0;index-- {
                if(nums[index] > nums[i-1]) {
                    break
                }
            }
            nums[i-1],nums[index] = nums[index], nums[i-1]
            reverse(i, length-1)
            break
        }
    }
    if flag == 0 {
        reverse(0, length-1)
    }
}
```

### 50. 缺失的第一个正数

[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

空间O(n)的比较容易想到，怎么优化为空间O(1)？

```go
// 空间O(n)
func firstMissingPositive(nums []int) int {
    length := len(nums)
    cnt := make([]int, length+1)
    for i:=0; i <length;i++ {
        if nums[i] >0 && nums[i] <= length {
            cnt[nums[i]] = 1
        }
    }
    for i :=1; i<=length;i++ {
        if cnt[i] != 1 {
            return i
        }
    }
    return length+1
}

//空间O(1)
func firstMissingPositive(nums []int) int {
    // 思路是，没有出现的最小正整数只有2种情况：
        // - 是 1~len(nums)中的一个数
        // - 或者是 len(nums)+1
    // 因为nums中的数字最多把1~len(nums)中的每个数都包含
    // 所以我们只需要把nums中在1~len(nums)之间的数放到原数组的对应位置，最后下标与数值不等的下标就是所求
    // 这里说的放到对应的位置用的是交换两个值，直到无法交换为止（即该值不在1~len(nums)之间）
    for _, n := range nums {
        for ; n >= 1 && n <= len(nums) && nums[n-1] != n; {  //这里要把n存到nums[n-1]的位置上，以为这样才能保存n==lenght时的n
            n, nums[n-1] = nums[n-1], n
        }
    }
    for index, n := range nums{
        if index+1 != n {
            return index+1
        }
    }
    return len(nums) + 1
}
```

### 51. 括号生成

[22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```go
func generateParenthesis(n int) []string {
    res := make([]string, 0)
    var dfs func (left int, right int, str string)
    dfs = func(left int,  right int, str string) {
        if len(str) == n * 2 {
            res = append(res, str)
            return
        }
        if left < n {
            dfs(left+1, right, str+"(")
        }
        if right < left {
            dfs(left, right+1, str+")")
        }
    }
    dfs(0,0,"")
    return res
}
```

### 52. 二叉树前序遍历

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)



```go
// 递归
func preorderTraversal(root *TreeNode) []int {
    var res []int
    var dfs func(root *TreeNode)
    dfs = func(node *TreeNode) {
        if node == nil {
            return
        }
        res = append(res, node.Val)
        dfs(node.Left)
        dfs(node.Right)
    }
    dfs(root)
    return res
}

//迭代
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode *> stk;
        vector<int> res;
        if(root == nullptr) return res;
        stk.push(root);
        while(!stk.empty() ){
            TreeNode *node = stk.top();
            stk.pop();
            res.push_back(node->val);
            if(node->right) stk.push(node->right);
            if(node->left) stk.push(node->left);
        }
        return res;
    }
};
```



### 53. 颠倒字符串中的单词

[151. 颠倒字符串中的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

```c++
class Solution {
public:
    string reverseWords(string s) {
        removeSpace(s);
        reverseStr(s, 0, s.size()-1);
        int l = 0;
        for(int i = 0;i < s.size();i++) {
            if(s[i] == ' ') {
                reverseStr(s, l, i-1);
                l = i + 1;
            }
        }
        reverseStr(s, l , s.size()-1);
        return s;
    }

    void reverseStr(string &s, int l, int r) {
        for(int i = l;i <= (r+l) / 2;i++) {
            swap(s[i], s[r+l-i]);
        }
    }
    
	// 思路是把经过筛选的字符从头开始放置，最后截取多余的字符（str.resize函数）
    void removeSpace(string &s) {
        int len = s.size();
        int fast = 0;
        int slow = 0;
        while(fast < len && s[fast] == ' ') {
            fast++;
        }
        while(fast < len) {
            while(fast < len && s[fast] != ' ') { //注意这里的fast<len
                s[slow++] = s[fast++];
            }
            if(slow < s.size()-1 && fast < s.size()-1)  // 注意这里的slow和fast的判断，是针对开头结尾没有空格的情况
                s[slow++] = s[fast++];
            while(fast < len && s[fast] == ' ') {
                fast++;
            }
        }
        if(slow > 0 && s[slow-1] == ' ') {
            s.resize(slow-1);
        }else s.resize(slow);
    }
    // void removeSpace(string &s) {
    //     int len = s.size();
    //     int slow = 0;
    //     int fast = 0;
    //     while(fast < s.size() && s[fast] == ' '){
    //         fast++;
    //     }
    //     while(fast < len) {
    //         if (fast > 0 && s[fast] == ' ' && s[fast] == s[fast-1]) {
    //             fast++;
    //         }else {
    //             s[slow++]=s[fast++];
    //         }
    //     }
    //     // cout << slow << endl;
    //     if(slow > 0 && s[slow-1] == ' ') {
    //         s.resize(slow-1);
    //     }else {
    //         s.resize(slow);
    //     }
    // }
};
```

### 54. 复原 IP 地址

[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

有很多细节要注意

```c++
class Solution {
public:
    vector<string> res;
    vector<string> restoreIpAddresses(string s) {
        dfs(s, 0, 0);
        return res;
    }

    void dfs(string &s,int index,  int pointNum){
        if(pointNum == 3) {
            if(judge(s, index, s.size()-1)) {
                res.push_back(s);
                return;
            }
        }
        for(int i = index;i < s.size();i++) {
            if(judge(s, index, i)) {
                s.insert(s.begin()+i+1, '.');  //insert的参数的含义？   这里是在i的后面插入‘.’
                dfs(s, i+2, pointNum+1);   //注意这里是i+2
                s.erase(s.begin()+i+1);
            }
        }
    }

    bool judge(string s, int l, int r) { 
        if(l >= s.size()) return false;  // 保证最后一段的长度大于等于1
        int num = 0;
        for(int i = l;i <= r;i++) {
            if(num > 255) return false;   // 合规判断-这里写在前面，防止溢出；
            if(i == (l+1) && s[l] == '0'){
                return false;
            }
            num = num*10 + (s[i] - '0');
        }
        if(num > 255) return false; // 合规判断
        return true;
    }
};
```



### 54. 从前序与中序遍历序列构造二叉树

[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return build(preorder, 0, preorder.size()-1, inorder, 0, inorder.size() -1);
    }
    // 对于先序的结点在中序中的位置还是通过遍历找到的
    TreeNode *build(vector<int> &preorder, int pl, int pr, vector<int> &inorder, int il, int ir) {
        if(il > ir) return nullptr;
        TreeNode *node = new TreeNode(preorder[pl]);
        int left = 0;
        for(int i = il; i <= ir;i++) {
            if(inorder[i] == preorder[pl]) {
                left = i;
                break;
            }
        }
        node->left = build(preorder, pl+1, pl + left - il, inorder, il, left-1);
        node->right = build(preorder, pl + left - il + 1, pr, inorder, left+1, ir);
        return node;
    }
};
```



### 55. 滑动窗口最大值

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

```c++
class Solution {
public:
    // 递减的单调队列
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;
        vector<int> res;
        if(nums.size() == 1) return nums;
        dq.push_back(0);
        // 注意一些特殊情况的处理
        if(k == 1) res.push_back(nums[0]);
        for(int i = 1;i < nums.size();i++) {
            // 每一轮一定要放入当前元素，所以先对队列进行操作，然后入队列
            while(!dq.empty() && nums[i] > nums[dq.back()]) {
                dq.pop_back();
            }
            dq.push_back(i);
            // 入队列时候对队列长度进行判断，队列首元素的作用范围即为k
            if(!dq.empty() && i - dq.front() >= k) {
                dq.pop_front();
            }
            if(i >= k - 1) {
                res.push_back(nums[dq.front()]);
            }
        }
        return res;
    }
};
```



### 56. 最小覆盖子串

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

这题还是挺难的，主要是思路比较难想到

思路大体是从第一个匹配的字母开始，在s中找到能够完全匹配（t）的第一个字母，统计当前长度

然后l指针开始右移，直到排除掉一个t中的字母，然后r指针再开始右移去找该字母

用一个cnt变量来统计是否已经完全匹配了t（每匹配一个字母就+1， 直到cnt等于t的长度）

```c++
class Solution {
public:
    string minWindow(string s, string t) {
        map<char,int> mt;
        map<char,int> mh;
        for(auto c: t) {
            mt[c]++;
        }
        int l=0;
        int r = 0;
        int cnt = 0;
        int tlen = t.size();
        int slen = s.size();
        int res = slen+1;
        string str;
        while(l + tlen <= slen) {
            while(l + tlen <= slen && mt[s[l]] == 0) {
                l++;
            }
            while(r < slen && cnt < tlen) {
                if(mh[s[r]] < mt[s[r]]) {
                    cnt++;
                }
                mh[s[r]]++;
                r++;
            }
            if(cnt == tlen  && r - l < res) {
                res = r - l ;
                str = s.substr(l, r - l);
            }
            if(l + tlen <= slen) {
                if(mh[s[l]] <= mt[s[l]]) {
                    cnt --;
                }
                mh[s[l]]--;
                l++;
            }
        }
        return str;
    }
};
```



### 57. 求根节点到叶节点数字之和

[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

思路大体是使用先序遍历，每次把父节点的值乘10往下累加

到根节点的时候，把值累加到最后结果上即可

```c++
class Solution {
public:
    int sum = 0;
    int sumNumbers(TreeNode* root) {
        dfs(root, 0);
        return sum;
    }

    void dfs(TreeNode *node,int temp) {
        if(node == nullptr) return;
        if(node->left == nullptr && node->right == nullptr) {
            sum += temp*10 + node->val;
            return;
        }
        dfs(node->left, temp * 10+node->val);
        dfs(node->right, temp * 10+node->val);
    }
};
```



### 58. 平衡二叉树

[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

```go
func isBalanced(root *TreeNode) bool {
    res := true

    var dfs func(node *TreeNode) int
    dfs = func(node *TreeNode) int {
        if node == nil {
            return 0
        }else {
            left := dfs(node.Left)
            right := dfs(node.Right)
            if abs(left,right) > 1 {
                res = false
            }
            if left>right {
                return left+1
            }else {
                return right+1
            }
            // return left>right?left+1:right+1
        }
    }
    dfs(root)
    return res
}

func abs(a,b int) int {
    if a>b{
        return a - b
    }else {
        return b - a
    }
}
```



### 59. 二叉树的最大深度

[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        return dfs(root);
    }

    int dfs(TreeNode *root) {
        if(root == nullptr) return 0;
        int left = dfs(root->left);
        int right = dfs(root->right);
        return max(left,right) + 1;
    }
};
```



### 60. 最小栈

[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

使用一个链表，然后存储val已经当前位置的最小值即可

每次头插一个新节点

```c++
class MinStack {
public:
    typedef struct minstack{
        int val;
        int minval;
        struct minstack *next;
        minstack(int x):val(x), minval(INT_MAX),next(nullptr){}
    }ms;

    ms *s1;

    MinStack() {
        s1 = new minstack(0);
    }
    
    void push(int val) {
        ms *s2 = new minstack(val);
        if(val < s1->minval) {
            s2->minval = val;
        }else {
            s2->minval = s1->minval;
        }
        s2->next = s1;
        s1 = s2;
    }
    
    void pop() {
        s1 = s1->next;
    }
    
    int top() {
        return s1->val;
        // return 0;
    }
    
    int getMin() {
        // return 0;
        return s1->minval;
    }
};
```



### 61. 对称二叉树

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

```c++
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(root == nullptr) return true;
        return dfs(root->left, root->right);
    }

    bool dfs(TreeNode *left, TreeNode *right) {
        if(left == nullptr && right == nullptr) return true;
        if(left == nullptr || right == nullptr) return false;
        if(left->val == right->val) {
            return dfs(left->left, right->right) && dfs(left->right, right->left);
        }
        return false;
    }
};
```





### 62. 二叉树的直径

[543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

```c++
class Solution {
public:
    int res =  0;
    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        return res-1;
    }

    int dfs(TreeNode *node) {
        if(node == nullptr) return 0;
        int left = dfs(node->left);
        int right = dfs(node->right);
        res = max(res, left+right+1);
        return max(left,right)+1;
    }
};
```





### 63. 最长有效括号

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

这题的思路还是挺有意思的

采用动规，dp[i]表示以s[i]结尾的最长有效括号的长度

然后我们只需要对 ')' 做处理，对于每一个')'有两种情况：

- 前一个字符是'('，要再dp[i-2]的基础上+2
- 前一个字符是')'，要在dp[i-1]的基础上进行增加，需要由s[i - dp[i-1] - 1]来决定
    - 同时就是当s[i - dp[i-1] - 1]匹配成功，还要再考虑前面是否还能将断了的匹配括号连起来

```c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int len = s.size();
        vector<int> dp(len+1, 0);
        int res = 0;
        for(int i = 1;i < s.size();i++) {
            if(s[i] == ')'){
                if(s[i-1] == '(') {
                    if(i < 2) dp[i] = 2;
                    else dp[i] = dp[i-2] + 2;
                }else {
                    int index = i - dp[i-1] - 1;
                    if(index >= 0 && s[index] == '(') {
                        dp[i] = dp[i-1]+2;
                    }
                    if(index >= 0 && s[index] == '(' && index - 1 >= 0) {
                        dp[i] += dp[index-1];
                    }
                }
                res = max(res, dp[i]);
            }
        }
        return res;
    }
};
```



### 64. 验证二叉搜索树

[98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

递归，或者中序遍历为有序序列

```c++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return dfs(root, LLONG_MIN, LLONG_MAX);
    }
    bool dfs(TreeNode *root, long long minval, long long maxval) {
        if(root == nullptr) return true;
        if(root->val >= maxval || root->val <= minval) return false;
        return dfs(root->left,minval, root->val) && dfs(root->right, root->val, maxval);
    }
};
```



### 65. 路径总和II

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        dfs(root,targetSum);
        return res;
    }
    void dfs(TreeNode *root, int ret) {
        if(root == nullptr) {
            return;
        }
        if(root->left == nullptr && root->right == nullptr) {
            if(root->val == ret) {
                temp.push_back(root->val);
                res.push_back(temp);
                temp.pop_back();
            }
        } 
        temp.push_back(root->val);
        dfs(root->left, ret-root->val);
        dfs(root->right, ret-root->val);
        temp.pop_back();
    }

};
```



### 66. 比较版本号

[165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)

```c++
class Solution {
public:
    int compareVersion(string version1, string version2) {
        int i = 0;
        int j = 0;
        int len1 = version1.size();
        int len2 = version2.size();
        while(i < len1 || j < len2) {  // 这里注意是||，因为可能出现1.0.1 与 1.0
            int num1 = 0;
            int num2 = 0;
            while(i < len1 && version1[i] != '.') {
                num1 = num1 * 10 + (version1[i] - '0');
                i++;
            }
            while(j < len2 && version2[j] != '.') {
                num2 = num2 * 10 + (version2[j] - '0');
                j++;
            }
            if(num1 > num2) {
                return 1;
            }
            if(num1 < num2) {
                return -1;
            }
            i++;
            j++;
        }
        return 0;
    }
};
```

### 67. 字符串相乘

[43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)

直接每一位相乘，然后对应位上的数字直接累加

然后对每一位处理进位

最后转为字符串即可

```c++
class Solution {
public:
    string multiply(string num1, string num2) {
        int len = num1.size() + num2.size();
        vector<int> temp(len+1, 0);

        for(int i = num1.size()-1;i >=0;i--) {
            for(int j = num2.size()-1; j >=0 ;j--) {
                temp[i+j+1] += (num1[i]-'0') * (num2[j]-'0');
            }
        }
        int cnt = 0;
        for(int i = temp.size()-1; i >= 0; i--) {
            temp[i] += cnt;
            cnt = temp[i] / 10;
            temp[i] = temp[i] % 10;
        }
        int index = 0;
        while(index < temp.size() && temp[index] == 0) {
            index++;
        }
        string res = "";

        for(int i = index;i < len;i++) {
            res += to_string(temp[i]);
        }
        if(res.size() == 0) return "0";
        return res;
    }
};
```

### 68. 零钱兑换

[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

首先这里不能贪心，因为面额较大的并不一定要被选择



【dfs】

虽然会超时，但是思路很值得学习

这里要对每一种面额的每一种数量进行搜索



【dp】

```c++
class Solution {
public:
    int mincoin = INT_MAX;
    int coinChange(vector<int>& coins, int amount) {
        // 贪心的做法不对！如果总是先取面额最大的，可能会导致无解
        // 可使用 dfs 或者 dp

        //dfs -- 虽然会超时
        sort(coins.begin(), coins.end());
        dfs(coins, amount, coins.size()-1, 0);
        return mincoin == INT_MAX ? -1 : mincoin;
    }
    //dfs
    void dfs(vector<int> &coins,int amount,int index, int cnt) {
        if(index < 0 || cnt + amount/coins[index] > mincoin) {
            return;
        }
        if(amount % coins[index] == 0) {
            mincoin = min(mincoin, cnt + amount/coins[index]);
            return;
        }
        for(int i = amount/coins[index]; i >= 0;i--) {
            dfs(coins, amount - i * coins[index], index-1, cnt + i);
        }
    }
}; 


class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        // 贪心的做法不对！如果总是先取面额最大的，可能会导致无解
        // 可使用 dfs 或者 dp

        // dp[i][j]   前i个硬币凑成j所需要的最少数量
        vector<vector<int>> dp(coins.size()+1, vector<int>(amount+2, amount+1));
        int len = coins.size();

        dp[0][0] = 0
        // 这里为什么从i=1表示前1个硬币，而不是i=0表示前0个硬币？
        // 因为dp[i][j]要用到[i-1]，
        // 如果i=0表示前1个硬币，那么i=1就是从第2个硬币开始，那么我就要初始化所有的i=0的情况，但是i=0是和i=1一样的逻辑，并不能特殊处理
        // 如果i=1表示前1个硬币，那么用到i=0的时候，就表示0个硬币，那么其实初始化就变得很简单，因为没有硬币都是0
        for(int i = 1;i <= coins.size();i ++) {
            for(int j = 0; j <= amount;j++) {
                dp[i][j] =dp[i-1][j];
                if(j >= coins[i-1]) dp[i][j] = min(dp[i][j], dp[i][j-coins[i-1]] + 1);
            }
        }
        return dp[len][amount] == amount+1 ? -1 : dp[len][amount];
    }
}; 

// 空间优化版
class Solution {
public:
    int mincoin = INT_MAX;
    int coinChange(vector<int>& coins, int amount) {
        // 贪心的做法不对！如果总是先取面额最大的，可能会导致无解
        // 可使用 dfs 或者 dp

        int len = coins.size();
        vector<int> dp(amount+2, amount+1);
        dp[0] = 0;
        
        for(int i = 1;i <= coins.size();i ++) {
            for(int j = coins[i-1]; j <= amount;j++) {
                dp[j] = min(dp[j], dp[j-coins[i-1]] + 1);
            }
        }
        return dp[amount] == amount+1 ? -1 : dp[amount];
    }
    
}; 
```

### 69. 子集

[78. 子集](https://leetcode-cn.com/problems/subsets/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(nums, 0);
        return res;
    }
    void dfs(vector<int> &nums, int index) {
        res.push_back(temp);
        for(int i = index;i < nums.size();i++) {
            temp.push_back(nums[i]);
            dfs(nums, i+1);
            temp.pop_back();
        }
    }
};
```

### 69. 用 Rand7() 实现 Rand10()

[470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)

主要的思路是：

随机生成0-4的一个数（概率相等）

然后生成一个数是的奇偶概率相等（都为1/2）

用1/2的概率选0-4或者0-4 +5

```c++
class Solution {
public:
    int rand10() {
        int a = rand7();
        int b = rand7();
        while(a == 7) {  // a > 2也可以，但是会慢
            a = rand7();
        }
        while(b > 5) {
            b = rand7();
        }
        return a % 2 == 1 ? b : b + 5;
    }
};
```

### 70. 回文链表

[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

```c++
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        // 找到中间位置，然后反转后面一部分；最后对比即可
        if(head->next == nullptr) return head;
        int len = 0;
        ListNode *slow = head;
        ListNode *fast = head;
        while(fast) {
            fast = fast->next == nullptr ? fast->next : fast->next->next;
            slow = slow->next;
        }
        // reverse
        ListNode *pre = nullptr;
        while(slow) {
            ListNode *temp = slow->next;
            slow->next = pre;
            pre = slow;
            slow = temp;
        }
        while(pre && head) {
            if(pre->val != head->val) {
                return false;
            }
            pre=pre->next;
            head=head->next;
        }
        return true;
    }
};
```



### 71. 最小路径和

[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        int dp[m+1][n+1];
        dp[0][0] = grid[0][0];

        for(int i = 1;i < m;i++) dp[i][0] = grid[i][0] + dp[i-1][0];
        for(int i = 1;i < n;i++) dp[0][i] = grid[0][i] + dp[0][i-1];

        for(int i = 1;i < m;i++) {
            for(int j = 1;j < n;j++) {
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }
};
```



### 72. 多数元素

[169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int maxcnt = 1;
        int res = nums[0];
        for(int i = 1;i < nums.size();i++) {
            if(maxcnt == 0) {
                res = nums[i];
                maxcnt = 1;
                continue;
            }
            if(nums[i] == res) {
                maxcnt++;
            }else {
                maxcnt--;
            }
        }
        return res;
    }
};
```



### 73. 路径总和

[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

```c++
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        return dfs(root, targetSum);
    }
    // 单边遍历法更快？
    bool dfs(TreeNode *root, int ret) {
        if (root == nullptr) return false;
        if(root->left == nullptr && root->right == nullptr) {
            if(root->val == ret) {
                return true;
            }
        }
        bool l = dfs(root->left, ret - root->val);
        if (l) return true;
        bool r = dfs(root->right, ret - root->val);
        if (r) return true;     
        return false;
    }
};
```



### 74. 最长重复子数组

[718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

```c++
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        //子序列默认不连续，子数组默认连续
        int len1 = nums1.size();
        int len2 = nums2.size();
        int res = 0;
        vector<vector<int>> dp(len1+1, vector<int>(len2+1,0));
		//dp[i][j]表示以i，j结尾的最长重复子数组的长度
        for(int i = 0;i < len1;i++) {
            if(nums1[i] == nums2[0]) {
                dp[i][0] = 1;
                res = 1;
            }
        }
        for(int j = 0;j < len2;j++) {
            if(nums1[0] == nums2[j]) {
                dp[0][j] = 1;
                res = 1;
            }
        }
        for(int i = 1; i < len1; i++) {
            for(int j = 1;j < len2;j++) {
                // dp[i][j] = max()    这里不相等的时候，以ij结尾的最长重复子数组就为0，不用计算
                if(nums1[i] == nums2[j]) {
                    dp[i][j] = max(dp[i][j] , dp[i-1][j-1]+1);
                }
                res = max(res, dp[i][j]);
            }
        }
        return res;
    }
};
```



### 75. 组合总和

[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;	
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dfs(candidates, target, 0);
        return res;
    }
    void dfs(vector<int>& candidates, int ret,int index){
        if(ret < 0) return;
        if(ret == 0) {
            res.push_back(temp);
            return;
        }
        for(int i = index;i < candidates.size();i++) {
            temp.push_back(candidates[i]);
            dfs(candidates, ret - candidates[i], i);
            temp.pop_back();
        }
    }

};
```



### 76. 旋转图像

[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

先对折（左右或者上下都可），再对角

```c++
// 先左右对折
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        for(int i = 0;i < m;i++) {
            for(int j = 0;j < n/2;j++) {
                swap(matrix[i][j], matrix[i][n-1-j]);
            }
        }
        for(int i = 0;i < m;i++) {
            for(int j = 0; j < n-i;j++) {
                swap(matrix[i][j], matrix[m-j-1][n-i-1]);
            }
        }
    }
};

// 先上下对折
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        // 
        int m = matrix.size();
        int n = matrix[0].size();

        for(int i = 0; i < n/2;i ++) {
            swap(matrix[i], matrix[n-i-1]);
        }

        for(int i = 0;i < m;i++) {
            for(int j = 0;j <= i;j++) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};
```

### 77. 翻转二叉树

[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        dfs(root);
        return root;
    }
    void dfs(TreeNode *root) {
        if(root == nullptr) return;
        swap(root->left, root->right);
        dfs(root->left);
        dfs(root->right);
    }
};
```



### 78. 在排序数组中查找元素的第一个和最后一个位置

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        // 两次二分
        int len = nums.size();
        if(len == 0) return {-1,-1};
        vector<int> res;
        
        int l = 0;
        int r = len - 1;
        // start -- right = mid
        while(l < r) {
            int mid = (l+r) /2;
            if(nums[mid] >= target) {
                r = mid;
            }else {
                l = mid+1;
            }
        }

        if(nums[l] == target) {
            res.push_back(l);
        }else {
            return {-1,-1};
        }

        // end
        l = 0;
        r = len-1;
        while(l < r) {
            int mid = (l+r+1) / 2;
            if(nums[mid] <= target) {
                l = mid;
            }else {
                r = mid -1;
            }
        }
        if(nums[l] == target) {
            res.push_back(l);
        }else {
            return {-1,-1};
        }
        return res;
    }
};
```

### 79. 最长公共前缀

[14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

这里的思路其实是按照字典序排序之后，直接和最后一个字符串比较即可

```c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        sort(strs.begin(), strs.end());
        string res = "";
        int i = 0, j =0;
        while(i < strs[0].size() && j < strs[strs.size()-1].size()) {
            if(strs[0][i] == strs[strs.size()-1][j]) {
                res += strs[0][i];
                i++;
                j++;
            }else {
                break;
            }
        }
        return res;
    }
};
```

### 80. 删除排序链表中的重复元素

[83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

这里要注意的是，注释的那里应该是while，而不是if！！！！

```c++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode *dummy = new ListNode(0);
        dummy->next = head;

        while(head && head->next) {
            while(head && head->next && head->val == head->next->val) {  // 注意这里是while，不是if
                head->next = head->next->next;
            }
            if(head->next)
                head = head->next;
        }

        return dummy->next;
    }
};
```

### 81. 寻找峰值

[162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

由于最两边是负无穷大，所以有较大的数的那一边一定包含所求的结果

```c++
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int len = nums.size();
        int left = 0;
        int right = nums.size()-1;

        while(left < right) {
            int mid = (left+right)/2;
            if(mid + 1 < len &&  nums[mid] < nums[mid+1]) {
                left = mid + 1;
            }else {
                right = mid;
            }
        }
        return left;
    }
};
```



### 82. 不同路径

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

这题重要的是使用【组合数学】的方法进行求解

一共要走 m-1 + n-1步，其中m-1步向下，n-1步向左

我们只需要从 m-1+n-1中跳出m-1步向下（表示在第几步的时候向下），其他的都向左即可

问题就是：

- 组合数怎么求？

```c++
// dp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> dp(n+1,1);
        for(int i = 1;i < m;i++) {
            for(int j = 1;j < n;j++) {
                dp[j] = dp[j] + dp[j-1];
            }
        }
        return dp[n-1];
    }
};

//组合数
class Solution {
public:
    int uniquePaths(int m, int n) {
        // m+n-2   m-1
        int sum = m+n-2;
        int part = m-1;
        long long res = 1;
        for(int i = 1; i< m;i++) {
            res = res * (n+i-1) / i;
        }
        return res;
    }
};
```



### 83. 最长连续序列

[128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

【hash法】

先把所有数用hash保存一遍，然后对每个数寻找从他开始的连续的数

**优化**的点就是，如果比这个数小1的数也存在的话，那就不用从该数开始查找（减少查找次数）



【并查集】

用map来存储数据，按某个规则合并集合，然后就可以求解最终的结果

```c++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        // map & unionFind
        // map
        map<int,int> m;
        int len = nums.size();
        int res = 0;
        for(int i = 0;i < len;i++) {
            m[nums[i]] = 1;
            res = 1;
        }
        for(int i = 0;i < len;i++) {
            int cur = nums[i]+1;
            if(m.find(nums[i]-1) == m.end()) {  // 注意这里的优化
                if(m.find(cur) != m.end()) {
                    int curlen = 1;
                    while(m.find(cur) != m.end()) {
                        curlen++;
                        cur++;
                    }
                    res = max(res, curlen);
                }
            }
        }
        return res;
    }
};

// union Find
class Solution {
public:
    map<int,int> p;
    int longestConsecutive(vector<int>& nums) {
        // map & unionFind
        // unionFind
        int res = 0;
        int len = nums.size();
        for(int i = 0;i < len;i++) {
            p[nums[i]] = nums[i];
        }
        for(int i = 0;i < len;i++) {
            int a = nums[i];
            int b = nums[i]+1;
            if(p.find(b) != p.end()) {
                p[find(a)] = find(b);
            }
        }
        for(int i = 0;i < nums.size();i++) {
            int curlen = find(nums[i]) - nums[i]+1;   // 注意最后还要find一次
            res = max(res, curlen);
        }
        return res;
    }
    int find(int x) {
        if(p[x] != x) 
            p[x] = find(p[x]);
        return p[x];
    }
};
```

### 84. 搜索二维矩阵II

[240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        //从右上角看是一颗二叉搜索树
        int m = matrix.size();
        int n = matrix[0].size();
        bool res  = false;
        int i = 0;
        int j = n - 1;
        while(i < m && j >= 0) {
            if(matrix[i][j] == target) {
                res = true;
                break;
            }else if(matrix[i][j] < target) {
                i++;
            }else {
                j--;
            }
        }
        return res;
    }
};
```

### 85. 最大正方形

[221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

dp, 要注意的是取三者的最小值

```c++
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        int res = 0;
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        //init
        for(int i = 0;i < m;i++) {
            if(matrix[i][0] == '1') {
                dp[i][0] = 1;
                res = 1;
            }else {
                dp[i][0] = 0;
            }
        }
        for(int j = 0;j < n;j++) {
            if(matrix[0][j] == '1') {
                dp[0][j]=1;
                res = 1;
            }else {
                dp[0][j] = 0;
            }
        }
        //
        for(int i = 1;i < m;i++) {
            for(int j = 1;j < n;j++) {
                if(matrix[i][j] == '1') {
                    int len1 = dp[i-1][j];
                    int len2 = dp[i][j-1];
                    int len3 = dp[i-1][j-1];
                    dp[i][j] = min(len1, min(len2,len3))+1;  // 这里是三者的最小值？？？
                    res = max(res, dp[i][j]);
                }
                
            }
        }
        return res*res;
    } 
};
```



### 86. 寻找旋转排序数组中的最小值

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

是二分，但是是变形的二分

因为这里没有target，所以nums[mid]要和谁进行比较呢？

是mid+1， mid-1

还是l， r呢？？？

```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int len = nums.size();
        int l = 0;
        int r = len - 1;
        while(l < r) {
            int mid = (l+r) /2;
            if(nums[mid] < nums[r]) {  // 这里是和r进行比较
                r = mid;
            }else {
                l = mid + 1;
            }
        }
        return nums[l];
    }
};
```



### 87. 岛屿的最大面积

最直接的思路就是dfs，遍历这个岛屿，求出每一块的面积

也可以使用并查集，但是要做一些改动，具体的改动包括：

- 在进行集合合并的时候，要记录当前集合的面积（集合中结点的数量）
    - 具体的记录方式，我们可以使用：把二位的坐标换算成一维的，然后每合并一个结点，就向父节点（坐标更大的结点）累加

```c++
// dfs
class Solution {
public:
    int m,n,a;
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        if(grid.size() == 0) return 0;
         m = grid.size();
         n = grid[0].size();
        int res = 0;
        for(int i = 0;i <m;i++ ) {
            for (int j = 0;j < n;j++) {
                if(grid[i][j] == 1) {
                    a = 0;
                    dfs(grid, i, j);
                    res = max(res, a);
                }
            }
        }
        return res;
    }
    void dfs(vector<vector<int>> &grid, int x, int y) {
        if(x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0) 
            return;
        a++;
        grid[x][y] = 0;
        dfs(grid, x, y+1);
        dfs(grid, x+1, y);
        dfs(grid, x-1, y);
        dfs(grid, x, y-1);
    }
};
```



### 88. 字符串解码

[394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)

```c++
class Solution {
public:
    string decodeString(string s) {
        int i = 0;
        string res = decode(s, i);
        return res;
    }
    string decode(string &s, int &pos) {
        string t = "";
        int num = 0;
        int len = s.size();
        while(pos < len) {
            if(s[pos] == ']') {
                pos++;
                return t; 
            }else if(s[pos] <= '9' && s[pos] >= '0') {
                num = num * 10 + (s[pos] - '0');
                pos++;
            }else if(s[pos] == '[') {
                pos++;
                string cur = decode(s,pos);
                while(num--) {
                    t += cur;
                }
                // pos++;
                num = 0;   // 这个很重要！！！！！！
            }else { // abcdefg
                t += s[pos];
                pos++;
            }
        }
        return t;
    }
};
```

### 89. 排序数组-堆排

[912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

这里使用的堆排序-升序

这里和一般的堆排序的模板不同，因为是在原数组上对数据排序，而不是直接把数据输出

所以我们采用的先生成大顶堆，然后每次把当前长度内的最大值放到当前长度的最后一个位置，然后重新调整堆

```c++
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        int len = nums.size();
        for(int i = len/2; i >= 0;i--) {
            down(nums, i, len);
        }
        for(int i = len-1;i >= 0;i--) {
            swap(nums[0], nums[i]);
            down(nums, 0, i);
        }
        return nums;
    }
    void down(vector<int> &nums, int x, int len) {
        int t = x;
        if(x*2 + 1 < len && nums[t] < nums[x*2+1]) {
            t = x*2+1;
        }
        if(x * 2+2 < len&& nums[t] < nums[x*2+2]) {
            t = x*2+2;
        }
        if(t != x) {
            swap(nums[t], nums[x]);
            down(nums, t, len);
        }
    }
};
```

### 90. 两两交换链表中的结点

[24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        head = swapTwo(head);
        return head;
    }

    ListNode* swapTwo(ListNode *head) {
        if(head == nullptr) return head;
        ListNode *p = head;
        ListNode *q = head->next;
        if(q == nullptr) return p;
        ListNode *temp = q->next;
        q->next = p;
        p->next=swapTwo(temp);
        return q;
    }
};
```

### 91. 基本计算器II

[227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

在最前面加一个+，每次记录s[i]的前一个运算符

- 当s[i]是运算符的时候对前一个运算符进行运算（*/），加减法都直接入栈（减法入栈负数即可）
    - 这里还有就是当时最后一个字符的时候也要对前面的运算符进行运算

- 当s[i]是数字的时候记录数字

最后计算栈中的数字就好（都是加法）

```c++
class Solution {
public:
    int calculate(string s) {
        char pre = '+';
        int num = 0;
        stack<int> st;

        int index = 0;
        int len = s.size();

        for(int i = 0;i < len;i++) {
            if(s[i] <= '9' && s[i] >= '0') {
                num = num * 10 + (s[i]-'0');
            }
            if((s[i]=='*' || s[i] == '/' || s[i] == '+' || s[i] == '-') || i == len - 1){ // ！！！
                if(pre == '+') {
                    st.push(num);
                }else if(pre == '-') {
                    st.push(-num);
                }else {
                    int temp = pre == '*' ? st.top() * num : st.top() / num;
                    st.pop();
                    st.push(temp);
                }
                pre = s[i];
                num = 0;
            }
        }
        int res = 0;
         while(!st.empty()) {
            res += st.top();
            st.pop();
        }
        return res;
    }
}; 
```



### 92. 乘积最大子数组

[152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

```c++
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int len = nums.size();
        int res=nums[0];
        vector<int> mindp(len, 0);
        vector<int> maxdp(len,0);
        mindp[0] = nums[0];
        maxdp[0] = nums[0];

        for(int i = 1; i < len;i++) {
            if(nums[i] < 0) {
                mindp[i] = min(nums[i], maxdp[i-1] * nums[i]);
                maxdp[i] = mindp[i-1] * nums[i];
            }else {
                mindp[i] = mindp[i-1] * nums[i];
                maxdp[i] = max(maxdp[i-1] * nums[i], nums[i]);
            }
            res=max(res, maxdp[i]);
        }
        return res;
    }
};
```



### 93. 二叉树最大宽度

[662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/)

这题主要是要计算空结点，就是说是计算当前层的最右边节点和最左边节点的距离差（包含两个节点之间的空结点）

使用的方法是利用完全二叉树的下标的性质，当前结点的index=父节点的index*2（+1）

为了防止溢出，可以减去上层的第一个结点的index，即为start（或者start*2也可以）

最后计算差值即可

```c++
class Solution {
public:
    int widthOfBinaryTree(TreeNode* root) {
        if(root == nullptr) return 0;
        queue<pair<TreeNode*,long long>> q;
        q.push({root,1});
        long long res = 0;
        while(!q.empty()) {
            int len = q.size();
            long long start = q.front().second;
            long long index = 0;
            while(len--) {
                TreeNode *node = q.front().first;
                index = q.front().second;
                q.pop();
                if(node->left) {
                    q.push({node->left, (long long)index * 2  - start});
                }
                if(node->right){
                    q.push({node->right, (long long)index * 2 + 1 - start});
                }
            }
            res = max((int)res, (int)(index - start + 1));
        }
        return res;
        
    }
};
```

### 94. 只出现一次的数字

[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        // 时间O(n), 空间O(1)
        // 使用异或实现
        int res = 0;
        for(auto n:nums) {
            res = res ^ n;
        }
        return res;
    }
};
```

### 95. 验证IP地址

 [468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)

注意过滤前缀0的方式

```c++
class Solution {
public:
    string validIPAddress(string queryIP) {
        if (isIPv4(queryIP)) {
            return "IPv4";
        } else if (isIPv6(queryIP)) {
            return "IPv6";
        } else {
            return "Neither";
        }
    }
    bool isIPv4(string ip) {
        int num = 0;
        char pre;
        int cnt = 0;
        int len = ip.size();
        int i = 0;
        while(i < len) {
            if(ip[i] <= '9' && ip[i] >= '0') {
                if(pre == '0' && num == 0) return false;   // 前缀0的过滤方式
                num = num * 10 + (ip[i]-'0');
                if(num > 255 || num < 0) return false;
                pre = ip[i];
            }
            else if(ip[i] == '.') {
                if(pre == '.') return false;
                pre = ip[i];
                num = 0;
                cnt++;
            }else {
                return false;
            }
            i++;
        }
        cnt++;
        if(pre == '.') return false;
        if(cnt != 4) return false;
        if(num > 255 || num < 0) return false;
        return true;
    }

    bool isIPv6(string ip) {
        char pre;
        int cnt = 0;
        int perlen = 0;
        int len = ip.size();
        int i = 0;
        while(i < len) {
            if((ip[i] <= '9' && ip[i] >= '0') || (ip[i] <= 'f' && ip[i] >= 'a') || (ip[i] <= 'F' && ip[i] >= 'A')) {
                perlen++;
                pre = ip[i];
            }
            else if(ip[i] == ':') {
                if(perlen > 4) return false;
                if(pre == ':') return false;
                pre = ip[i];
                perlen = 0;
                cnt ++;
            }else {
                return false;
            }
            i++;
        }
        cnt++;
        if(cnt != 8) return false;
        if(pre == ':') return false;
        if(perlen > 4) return false;
        return true;
    }
};
```



### 96. 买卖股票的最佳时机II

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res = 0;
        for(int i = 0; i < prices.size()-1;i++) {
            if(prices[i+1] > prices[i]) {
                res += prices[i+1] - prices[i];
            }
        }
        return res;
    }
};
```

 

### 97. 打家劫舍

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

dp[i]表示当前偷用户时的最大收益

可以由前一个用户未被偷 以及 前两个用户被偷+当前收益 这两种状态转移而来

注意，由且仅由

还有一点就是初始化问题，就是dp[1] = max(nums[0], nums[1])

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        int len = nums.size();
        if(len == 1) return nums[0];
        vector<int> dp(len+1, 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for(int i = 2;i < len;i++) {
            dp[i] = max(dp[i-1] ,dp[i-2] + nums[i]);
        }
        return max(dp[len-1], dp[len-2]);
    }
};
```

### 98. 二叉搜索树与双向链表

[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

这里是按照中序遍历的结果进行双向链表的处理

我们按照中序遍历进行处理

对于中序遍历中的每个结点，我们处理它和它前一个结点(pre)的双向关系，即pre的right和当前root的left即可

我们需要记录pre结点，遍历结束时pre为最后一个节点，因此我们还需要记录第一个节点head（中序遍历当pre为空时root即为第一个结点）

然后在遍历完成之后处理第一个和最后一个结点的关系

```c++
class Solution {
public:
    Node *pre;
    Node *head;
    Node* treeToDoublyList(Node* root) {
        if(root == NULL) return NULL;
        dfs(root);
        head->left = pre;
        pre->right = head;
        return head;
    }
    void dfs(Node *root) {
        if(root == NULL) return;
        dfs(root->left);
        if(pre != NULL) {
            pre->right = root;
        }else {
            head = root;
        }
        root->left = pre;
        pre = root;
        dfs(root->right);
    }
};
```



### 99. 复制带随机指针的链表

[138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

[一] 可以使用map记录当前节点和随机结点的映射关系

[二] 在每个节点后面新建一个节点，这样先处理next关系，再处理random关系，最后分离两个链表即可

```c++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head == NULL) return NULL;

        Node *p = head;
        while(p) {
            Node *temp = new Node(p->val);
            temp->next = p->next;
            p->next = temp;
            p = p->next->next;
        }

        p = head;
        while(p) {
            if(p->random == NULL) {
                p->next->random = NULL;
            }
            else p->next->random = p->random->next;
            p = p->next->next;
        }

        Node *newhead = head->next;
        p = head;
        // Node *q = newhead;
        while(p) {
            // if(p->next != NULL)
            //     p->next = p->next->next;
            // if(q->next != NULL)
            //     q->next = q->next->next;
            // p = p->next;
            // q = q->next;
            Node *tmp = p->next;
            if(tmp)
                p->next = tmp->next;
            p = tmp;
        }
        return newhead;
    }
};
```

### 100. 最大数

[179. 最大数](https://leetcode-cn.com/problems/largest-number/)

主要是怎么比较两个数组？正确的比较方式就是a+b 和 b+a 拼接之后进行比较

```c++
class Solution {
public:
    static bool cmp (int a,int b) {
        string sa = to_string(a);
        string sb = to_string(b);
        return sa + sb > sb + sa;
    }
    string largestNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end(), cmp);
        string res = "";
        for(auto n : nums) {
            string t = to_string(n);
            if(n == 0 && res[0] == '0') continue;
            res += t;
        }
        return res;
    }
};
```

