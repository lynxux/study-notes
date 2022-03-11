### 1. 反转链表

[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```c++
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
            start = max(start, cnt[s[i]]);
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
    int res = 0;
    int findKthLargest(vector<int>& nums, int k) {
        // 快排或者并归排序
        return qs(nums, 0 ,nums.size() -1, k);

    }
    //quicksort
    int qs(vector<int> &nums, int left, int right, int k) {
        int p = part(nums, left, right);
        int m = right - p + 1;
        if(m == k) {
            return nums[p];
        }
        if(m < k) {
            return qs(nums, left, p-1, k - m);
        }else {
            return qs(nums, p+1, right, k);
        }
    }
    int part(vector<int> &nums, int left, int right) {
        int temp = nums[left];
        while(left<right){
            while(left<right&&nums[right]>=temp) right--;
            nums[left]= nums[right];
            while(left<right&&nums[left]<=temp) left++;
            nums[right]= nums[left];
        }
        nums[left] = temp;
        return left;
    }
};

//堆排序
class Solution {
public:
    int len;
    int findKthLargest(vector<int>& nums, int k) {
        len = nums.size();
        int temp = nums[len-1];
        for(int i = len - 1;  i >= 1; i --) {
            nums[i] = nums[i-1];
        }
        nums.push_back(temp);
        for(int i = len/2; i; i--) {
            down(i,nums);
        }
        int res = 0;
        while(k--){
            res = nums[1];
            nums[1] = nums[len--];
            down(1,nums);
        }
        return res;
    }
    void down(int x, vector<int> &nums) {
        int t = x;
        if(x * 2 <= len && nums[x*2] > nums[t]) t = x *2;
        if(x * 2+ 1 <= len && nums[x*2+1] > nums[t])  t=x*2+1;
        if(t != x){
            swap(nums[t],nums[x]);
            down(t, nums);
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

