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



### 5. K 个一组翻转链表

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
```

