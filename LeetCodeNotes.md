

## 写在前面

这次刷题按照【[代码随想录](https://www.programmercarl.com/)】的章节顺序来刷的，预计会有接近200道题。

---

[TOC]



## 数组

### 1.二分查找

[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

最主要的是要考虑 `left <= right` 以及 `left` ，`right`的赋值。

判断的标准是该值是否还有可能是最终的答案，如果不是，则就可以排除该值。

```go
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



### 2.移除元素

[27. 移除元素](https://leetcode-cn.com/problems/remove-element/)

这里只考虑最终长度内的数据，所以采用的是双指针（快慢指针），对需要被的移除的值用后面的值进行赋值。

问题在于如果快速，正确的写出算法。这里使用 `i`作为快指针！最终结束时是判断快指针！

```go
func removeElement(nums []int, val int) int {
    length := len(nums)
    res := 0
    for i := 0; i < length; i ++ {
        if nums[i] != val {
            nums[res] = nums[i]
            res++
        }
    }
    return res
}
```

### 3.有序数组的平方

[977. 有序数组的平方](https://leetcode-cn.com/problems/squares-of-a-sorted-array/)

也是双指针的思想，不过有个问题时从数组的头尾开始会比较方便的进行计算。

```go
func sortedSquares(nums []int) []int {
    left := 0
    right := len(nums) - 1
    res := make([]int, len(nums))
    cnt := len(nums) - 1
    for left <= right {
        if(abs(nums[left]) < abs(nums[right])) {
            res[cnt] = nums[right] * nums[right]
            right --
        }else {
            res[cnt] = nums[left] * nums[left]
            left ++
        }
        cnt--
    }
    return res
}

func abs(a int) int {
    if a < 0 {
        return -a
    }else {
        return a
    }
}
```



### 4.长度最小的子数组

[209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

滑动窗口，即也是双指针的一种。

这里需要搞清楚

- 如何移动窗口的结束位置？
- 如何移动窗口的起始位置？

以及算法到底要怎么实现。

```c
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int sum = 0;
        int i = 0;
        int length = nums.size();
        int sublength = 0;
        int result = 0x3fff;
        for (int j = 0;j < length; j ++) {
            sum += nums[j];
            while(sum >= target) {
                sublength = j - i + 1;
                result = min(result, sublength);
                sum -= nums[i++]; 
            }
        }
        if( result == 0x3fff) return 0;
        else return result;
    }
};
```



### 5.螺旋矩阵II

[59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

这里的主要是通过总结规律找到每次遍历的变量以及不变量。

还有如何找到循环的终止条件。

```c
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res(n, vector<int>(n,0));
        if (n == 0) return res;
        int u = 0;
        int d = n-1;
        int l = 0;
        int r = n-1;

        int cnt = 1;
        while(true) {
            for (int i = l;i <= r;i++) {
                res[u][i] = cnt ++;
            }
            u++;
            if(u > d) break;
            for(int i = u; i<= d;i ++){
                res[i][r] = cnt++;
            }
            r --;
            if(r <l) break;
            for(int i = r; i>= l; i--){
                res[d][i] = cnt++;
            }
            d--;
            if(d < u) break;
            for(int i = d;i >= u;i --){
                res[i][l] = cnt++;
            }
            l++;
            if(l > r) break;
        }
        return res;
    }
};
```



## 链表

### 1.移除链表元素

[203. 移除链表元素](https://leetcode-cn.com/problems/remove-linked-list-elements/)

还是双指针的思想（快慢指针），遍历一边链表即可

```c
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if (head == NULL)  return head;
        while(head->val == val && head->next != NULL) head = head->next;
        if(head ->val == val && head->next ==NULL) return NULL;
        ListNode *fast = head->next;
        ListNode *slow = head;
        while(fast != NULL){
            if(fast->val == val){
                while (fast != NULL && fast->val == val) {
                    fast = fast->next;
                }
                slow->next = fast;
            }else {
                slow = fast;
                fast = fast->next;
            }
        }
        return head;
    }
};
```



### 2.设计链表

[707. 设计链表](https://leetcode-cn.com/problems/design-linked-list/)

主要是有各种特殊情况需要考虑。

```c
class MyLinkedList {
public:
    struct LinkNode {
        int val;
        LinkNode *next;
        LinkNode (int val): val(val),next(nullptr){}
    };
    int size;
    LinkNode *head;

    MyLinkedList() {
        size = 0;
        head = NULL;    
    }
    
    int get(int index) {
        if(index > size - 1 || index < 0) return -1;
        int cnt = 0;
        LinkNode *p = head;
        while(cnt != index) {
            p = p->next;
            cnt ++;
        }
        return p->val;
    }
    
    void addAtHead(int val) {
        LinkNode *p = new LinkNode(val);
        p->next = head;
        head = p;
        size++;
        // cout << size << " ";
    }
    
    void addAtTail(int val) {
        LinkNode *p = new LinkNode(val);
        LinkNode *h = head;
        if(h == NULL) {
            addAtHead(val);
            return;
        }
        while(h->next != NULL) {
            cout << h->val;
            h = h->next;
        }
        h->next = p;
        size++;
    }
    
    void addAtIndex(int index, int val) {
        if(index < 0) {
            addAtHead(val);
            return;
        }
        if(index == 0) {
            addAtHead(val);
            return;
        }
        if(index > size) return;
        if(index == size) {
            addAtTail(val);
            return;
        }
        LinkNode *p = new LinkNode(val);
        LinkNode *h = head;
        int cnt = 0;
        index --;
        while(cnt != index) {
            cnt ++;
            h = h -> next;
        }
        p->next = h->next;
        h->next = p;
        size ++;
    }
    
    void deleteAtIndex(int index) {
        if(index < 0 || index >= size) return;
        if(index==0){
            head = head->next;
            size --;
            return;
        }
        int cnt = 0;
        index --;
        LinkNode *h = head;
        while(cnt != index) {
            cnt ++;
            h = h->next;
        }
        h->next = h->next->next;
        size--;
    }
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */
```



### 3.反转链表

[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

还是双指针，还是比较容易的。

```c
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        
        if (head == NULL || head->next == NULL) return head;
        ListNode *slow = NULL;
        ListNode *fast = head;
        while(fast != NULL){
            ListNode *temp = fast->next;
            fast->next = slow;
            slow = fast;
            fast = temp;
        }
        return slow;
    }
};
```



### 4.两两交换链表中的节点

[24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

1. 要不要加虚拟头结点？
    - 要加，否则对头节点就要特殊处理
2. 使用几个指针？一个还是两个还是三个?
    - 1个或者2个都可以，三个就没有必要了
3. 交换的步骤到底是什么？
4. 指针每次前进几步？

```c
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode *tempNode = new ListNode(0);
        tempNode->next = head;
        ListNode *s = tempNode;
        ListNode *f = head;
        while(f != nullptr && f->next != nullptr){
            ListNode *temp = f->next->next;
            s->next = f->next;
            f->next->next = f;
            f->next = temp;
            s = s->next->next;
            f = s->next;
        }
        return tempNode->next;
    }
};
```



### 5.删除链表的倒数第N个节点

[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

主要是需要使用一个指针先前进n步，另一个指针从head开始即可。

```c
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        int cnt = n;
        ListNode *p = head;
        while(cnt --){
            p = p->next;
        }
        if(p == nullptr) {
            head = head ->next;
            return head;
        }
        ListNode *s = head;
        while(p->next) {
            p = p->next;
            s = s->next;
        }
        s->next=s->next->next;
        return head;
    }
};
```



### 6.链表相交

[面试题 02.07. 链表相交](https://leetcode-cn.com/problems/intersection-of-two-linked-lists-lcci/)

长的链表先前进（长链表长度 - 短链表长度）步，再一起出发，相遇时即为相交结点。

```c
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        int lenA = 0;
        int lenB = 0;
        ListNode *pa = headA;
        ListNode *pb = headB;
        while(pa) {
            lenA++;
            pa = pa->next;
        }
        while(pb) {
            lenB++;
            pb = pb->next;
        }
        pa = headA;
        pb = headB;
        int ab = lenA - lenB;
        if(ab > 0) {
            while(ab--){
                pa=pa->next;
            }
        }else{
            while(ab != 0) {
                ab++;
                pb = pb->next;
            }
        }
        while(pa != pb){
            pa = pa->next;
            pb = pb->next;
        }
        return pa;
    }
};
```



### 7.环形链表II

[142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

之前做过，思路还记得，但是细节记不清楚了。

使用快慢指针，快指针每次走2步，慢指针每次走一步。

如果有环，则在**慢指针走完第一圈前，快慢指针必然会相遇**。并且，【相遇时所在结点】到【相切结点】的距离与【头结点】到【相切结点】的【距离相等】。

```c
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if (head == NULL) return NULL;
        ListNode *slow = head;
        ListNode *fast = head;
        while(fast != NULL && fast->next != NULL) {
            slow = slow->next;
            fast = fast->next->next;
            if(slow == fast) break;
        }
        if(fast == NULL ||fast->next == NULL) return NULL;
        fast = head;
        while(fast != slow){
            fast = fast->next;
            slow = slow->next;
        }
        return fast;
    }
};
```





## 哈希表

### 1.有效的字母异位词

[242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

可以考虑一下使用什么数据结构来存储每个字母出现的次数，需不需要使用map？

```c
class Solution {
public:
    bool isAnagram(string s, string t) {
        int record[27] = {0};
        for(int i = 0;i < s.size(); i++) {
            record[s[i]-'a'] ++;
        }
        for(int i = 0;i < t.size(); i++){
            record[t[i] - 'a']--;
        }
        for(int i = 0;i < 26;i++){
            if(record[i] != 0) return false;
        }
        return true;
    }
};
```



###  2.两个数组的交集

[349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

主要就是不重复的存一遍`nums1`的数据，再根据`nums2`的数据做判断。

主要还是数据结构的选择，这里也可以使用`unordered_set`

```c
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        vector<int> res;
        map<int, int> record1;
        for(int num: nums1){
            if(record1.count(num) != 0){
                continue;
            }else{
                record1.insert(make_pair(num, 1));
            }
        }
        for(int num:nums2){
            if(record1.count(num) != 0 && record1[num] != 0x3ff){
                res.push_back(num);
                record1[num] = 0x3ff;
            } 
        }
        return res;
    }
};
```



### 3.快乐数

[202. 快乐数](https://leetcode-cn.com/problems/happy-number/)

平方的和重复的话，即失败。

```c
class Solution {
public:
    int getSum(int n){
        int sum = 0;
        while(n){
            int a = n % 10;
            n = n / 10;
            sum += a * a;
        }
        return sum;
    }
    bool isHappy(int n) {
        unordered_set<int> record;
        while(1){
            int sum = getSum(n);
            if(sum == 1) return true;
            if(record.find(sum) != record.end()) return false;
            record.insert(sum);
            n =sum;
        }
        return true;
    }
};
```



### 4.两数相加

[1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

暴力的方法是很容易想到的，时间复杂度为$O(n^2)$

这里想要优化时间复杂度，使用的是哈希表`unordered_map`，使用`unordered_map`之后，把已经查询过的`num`放入`map`中，查询`target-nums[i]`是否在`map`中。

查询`target-nums[i]`这一步本来是$O(n)$，使用`map`之后减少为$O(1)$

```c
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> re;
        for(int i = 0; i< nums.size();i++){
            int remain = target - nums[i];
            if(re.find(remain) != re.end()){
                return vector<int>{i,re[remain]};
            }else {
                re[nums[i]] = i;
            }
        }
        return {-1,-1};
    }
};
```



### 5.四数相加II

[454. 四数相加 II](https://leetcode-cn.com/problems/4sum-ii/)

暴力求解是四层循环，使用`unordered_map`可以减少一次循环，每两个数组求和作为一个`map`，可以减少2层循环

 ```c
class Solution {
public:
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4) {
        unordered_map<long,long> m1;
        unordered_map<long,long> m2;
        int n = nums1.size();
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n;j++) {
                m1[nums1[i] + nums2[j]] ++;
                m2[nums3[i] + nums4[j]] ++;
            }
        }
        unordered_map<long,long>::iterator it;
        int sum = 0;
        for(it = m1.begin(); it != m1.end(); it ++) {
            if(m2.find(-1 * it->first) != m2.end()){
                sum += (m2[-1 * it->first]*it->second);
            }
        } 
        return sum;
    }
};
 ```



### 6.赎金信

[383. 赎金信](https://leetcode-cn.com/problems/ransom-note/)

```c
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        if (ransomNote.length() > magazine.length()) return false;
        unordered_map<int,int> record;
        for(int i = 0; i < magazine.length(); i++){
            record[magazine[i]]++;
        }
        for(int i = 0; i<ransomNote.length(); i++){
            record[ransomNote[i]]--;
            if(record[ransomNote[i]] < 0) return false;
        }
        return true;
    }
};
```



### 7.三数之和

[15. 三数之和](https://leetcode-cn.com/problems/3sum/)

比起之前的两数之和以及四数相加，这里多了一个要求就是不能出现重复的组合。

所以最主要的问题就是如何去重。

简单的想法的就是对不同的元素为第一个元素进行组合的求解，但是又不能直接把数组去重，因为虽然每个数字只能用一次，但是可能存在相同的数字。

所以**第一步**就先排序，相同的元素只求解一次。

然后就考虑如何进行第二个数和第三个数的筛选。

这里是使用**双指针**进行求解，对于第二个数我们的要求也是一样的，即多个相同的元素，只选中一次，然后进行第三个元素的求解。（**注意**，这里的第二个元素是可以和第一个元素相等的，但不能是同一个元素。所以第二个元素只需要从第一个元素的下一个开始即可。）

第三个元素也是一样的，多个相同的元素，也只选择一次作为最终的答案。那么第三个元素从哪里开始呢？第三个元素也是可以从第二个元素的下一个元素开始，但是这样的话第二个和第三个元素的的查找时间为$O(n^2)$，总的为$O(n^3)$.

如果我们第三个元素从最后一个元素向前查找，那么第二个和第三个元素的的查找时间为$O(n)$，总的为$O(n^2)$.

所以，查找第二个元素的指针向尾前进，查找第三个元素的指针向头前进，当第二次筛选及以后的筛选过程中，当前元素与上一轮的元素重复时，即直接跳过。

```c
class Solution {
public:
    //双指针更简单
    //这里主要是去重操作
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        int len = nums.size();
        if(len < 3 || nums[0] > 0 || nums[len-1] < 0) return res;
        for(int i = 0;i < len; i++){
            if(nums[i] > 0) break;
            if(i > 0 && nums[i] == nums[i-1]) continue;
            int target = 0 - nums[i];
            int s = i + 1;
            int f = len - 1;
            while(s < f) {
                if(nums[s] > target) break;
                if(( nums[s] == nums[s-1] && s > i+1 ) || nums[s] + nums[f] < target) s++;
                else if((f < len -1 && nums[f] == nums[f+1]) || nums[s] + nums[f] > target) f--;
                else {
                    vector<int> re;
                    re.push_back(nums[i]);
                    re.push_back(nums[s]);
                    re.push_back(nums[f]);
                    res.push_back(re);
                    s++;
                }
            }
        }
       return res;
    }
};
```



### 8.四数之和

[18. 四数之和](https://leetcode-cn.com/problems/4sum/)

这里还是三数之和的思想，只不过多一层循环求解。

值得注意的是，这里被注释掉的几个剪枝操作。

由于这里的target是固定的，所以，`nums[i] >targe`t时不能保证就没有可行解，比如`nums = [-4,-3,0,....], target=-7`

同理，`nums[len-1] < targe`时也不能保证就没有可行解，比如`nums = [....0,1,2,3], target=-6`

```c
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        int len = nums.size();
        // if(len < 4 || nums[0] > target || nums[len - 1] < target) return res;

        for(int i = 0;i < len; i++){
            // if(nums[i] > target) break;
            if(i > 0 && nums[i] == nums[i-1]) continue;
            for(int j = i + 1; j< len;j++){
                // if(nums[j] > target) break;
                if(j > i + 1 && nums[j] == nums[j-1]) continue;
                int target2 = target - nums[i] - nums[j];
                int s = j + 1;
                int f = len - 1;
                while( s < f) {
                    // if(nums[s] > target2) break;
                    if((s > j+1 && nums[s] == nums[s-1]) || nums[s] + nums[f] < target2 ) s++;
                    else if((f < len - 1 && nums[f] == nums[f+1]) || nums[s] + nums[f] > target2) f--;
                    else {
                        vector<int> re;
                        re.push_back(nums[i]);
                        re.push_back(nums[j]);
                        re.push_back(nums[s]);
                        re.push_back(nums[f]);
                        res.push_back(re);
                        s++;
                    }
                }
            }
        }
        return res;
    }
};
```



## 字符串

### 1.反转字符串

[344. 反转字符串](https://leetcode-cn.com/problems/reverse-string/)

```c
class Solution {
public:
    void reverseString(vector<char>& s) {
        int len = s.size();
        for(int i = 0;i < len/2; i++){
            char temp = s[i];
            s[i] = s[len-i-1];
            s[len-i-1] = temp;
        }
    }
};
```



### 2.反转字符串II

[541. 反转字符串 II](https://leetcode-cn.com/problems/reverse-string-ii/)

```c
class Solution {
public:
    string reverseStr(string s, int k) {
        int len = s.length();
        if(k == 1) return s;
        for(int i = 0 ; i < len;i=i+2*k) {
            int t = i;
            int e = i + k;
            if(e > len) {
                e = len;
            }
            for(int j = t; j < (t+e)/2; j++) {
                char temp = s[j];
                s[j] = s[e-j-1 + i];
                s[e-j-1 + i] = temp;    
            }
            
        }
        return s;
    }
};
```



### 3.替换空格

[剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

```c
class Solution {
public:
    string replaceSpace(string s) {
        int len = s.length();
        string res = "";
        int cnt = 0;
        for(int i = 0; i < len;i ++){
            if(s[i] == ' '){
                res.insert(cnt++,"%");
                res.insert(cnt++,"2");
                res.insert(cnt++,"0");
            }
            else{
                string str;
                stringstream stream;
                stream << s[i];
                str = stream.str();
                res.insert(cnt,str);
                cnt++;
            }
        }
        return res;
    }
};
```



### 4.翻转字符串里的单词

[151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

最基本的方法就是先去掉头尾的空格（记录头尾，使用`substr`），然后再从后向前遍历，把遇到的每个单词，拼接到新的字符串上。

时间复杂度为`O(n)`, 空间为`O(n)`

如果想要空间复杂度为`O(1)`，则需要再去除空格时不能使用新的字符串。这里的方法为使用双指针（快慢指针），慢指针用于表明字符应该放置的新位置，快指针用于筛选应该保留的字符。

对于上面的方法，只取出空格是不够的，因为单词的顺序没有办法反转。所以可以在取出空格之后，第一步先反转整个字符串，再反转每个单词，就可以得到最终结果。

```c
class Solution {
public:
    void reverse(string &s,int start, int end){
        for(int i = start;i < (start + end) /2;i++){
            swap(s[i],s[end - i - 1 + start]);
        }
    }
    void removeSpace(string &s) {
        int slow = 0;
        int fast = 0;
        //前面的空格
        while(s.size() > 0 && fast <s.size() && s[fast] == ' '){
            fast++;
        }
        //去除中间的多余空格
        //但是这里对于结尾存在空格的情况，会把一个空格保留，slow会指向最后一个空格的下一个位置
        for(; fast < s.size(); fast++){
            if(fast - 1 > 0 && s[fast] == s[fast-1] && s[fast] == ' '){
                continue;
            }else {
                s[slow++] = s[fast];
            }
        }
        //对最后的空格做处理
        if(slow - 1 > 0 && s[slow - 1] == ' '){
            s.resize(slow - 1);
        }else{
            s.resize(slow);
        }

    }

    string reverseWords(string &s) {
        int len = s.length();
        removeSpace(s);
        reverse(s,0,s.size());
        for(int i = 0;i < s.size(); i++){
            int j = i;
            while(j < s.size() && s[j] != ' ') j++;
            reverse(s,i,j);
            i = j;
        }
        return s;  
    }
};
```





## 栈与队列

### 1.用栈实现队列

[232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

主要是怎么存储数据。

很容易想到一个栈用于存入数据，一个栈用于输出数据。

当输出数据时，输出栈为空则从输出栈中全部存入输出栈；否则直接存输出栈输出元素即可。

```c
class MyQueue {
public:
    stack<int> stackIn;
    stack<int> stackOut;
    MyQueue() {

    }
    void push(int x) {
        stackIn.push(x);
    }
    
    int pop() {
        if(stackOut.empty()){
            while(!stackIn.empty()) {
                stackOut.push(stackIn.top());
                stackIn.pop();
            }
        }
        int res = stackOut.top();
        stackOut.pop();
        return res;
    }
    
    int peek() {
        int res = this->pop();
        stackOut.push(res); 
        return res;
    }
    
    bool empty() {
        return stackIn.empty() && stackOut.empty();
    }
};
```



### 2.用队列实现栈

[225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)

这题用一个队列就可以实现栈，主要是要在入队列时，就把队列里的元素调整为正确的顺序（出队列再入队列即可）。

如果要用两个队列实现，则在出栈时，把其他元素放到另一个队列中

```c
class MyStack {
public:
    queue<int> q;
    MyStack() {

    }
    void push(int x) {
        int len = q.size();
        q.push(x);
        while(len--){
            int a = q.front();
            q.pop();
            q.push(a);
        }
    }
    int pop() {
        int x = q.front();
        q.pop();
        return x;
    }
    
    int top() {
        return q.front();
    }
    
    bool empty() {
        return q.empty();
    }
};
```



### 3.有效的括号

[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```c
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        int len = s.size();
        if (len % 2 == 1) return false;
        for(int i = 0;i < len; i++){
            if(s[i] == '(' || s[i] == '[' || s[i] == '{') {
                st.push(s[i]);
            }
            else if(s[i] == '}' || s[i] == ']' || s[i] == ')'){
                if(st.size() == 0) return false;
                if(st.top() != '(' && s[i] == ')'){
                    return false;
                }
                if(st.top() != '{' && s[i] == '}'){
                    return false;
                }
                if(st.top() != '[' && s[i] == ']'){
                    return false;
                }
                st.pop();
            }

        }
        return st.empty();
    }
};
```



### 4.删除字符串中的所有相邻重复项

[1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

思路很容易，`st.empty()`要注意一下，过程中也可能出现栈为空的情况。

```c
class Solution {
public:
    string removeDuplicates(string s) {
        if(s.length() == 0 ) return s;
        stack<char> st;
        int len = s.size();
        st.push(s[0]);
        for(int i = 1;i < len; i++){
            if(st.empty() || s[i] != st.top()){ //这里的st.empty()！！
                st.push(s[i]);
            }else{
                st.pop();
            }
        }
        string res = "";
        while(!st.empty()){
            res += st.top();
            st.pop();
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
```



### 5.逆波兰表达式求值

[150. 逆波兰表达式求值](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

用栈还是很容易解决的

```c
class Solution {
public:
    int toInt(string s){
        int len = s.length();
        int sum = 0;
        int minus = 1;
        for(int i = len - 1;i >= 0; i--){
            if(s[i] == '-') {
                minus  = -1;
                continue;
            }
            sum = sum + (s[i] - '0') * pow(10,(len - i - 1));
        }
        return minus * sum;
    }
    int evalRPN(vector<string>& tokens) {
        stack<int> st;
        int res = 0;
        int len = tokens.size();
        for(int i = 0; i< len; i++) {
            if(tokens[i] == "+"){
                int a = st.top();
                st.pop();
                int b = st.top();
                st.pop();
                st.push(a + b);
            }  
            else if(tokens[i] == "-"){
                int a = st.top();
                st.pop();
                int b = st.top();
                st.pop();
                st.push( b - a);
            }  
            else if(tokens[i] == "*"){
                int a = st.top();
                st.pop();
                int b = st.top();
                st.pop();
                st.push(a * b);
            }  
            else if(tokens[i] == "/"){
                int a = st.top();
                st.pop();
                int b = st.top();
                st.pop();
                st.push(b/a);
            }else {
                int num = toInt(tokens[i]);
                st.push(num);
            } 
            
        }
        return st.top();
    }
};
```



###  6.滑动窗口最大值

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

标准的单调队列问题，主要注意的事项见注释即可。

```c
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        //单调队列 ->使用双向队列，或者自己维护一个数组和头尾指针
        //求滑动窗口的最大值，需要维护一个单调递减的队列（队列头为最大值），为了保存第二大的值的信息
        //求滑动窗口的最小值，需要维护一个单调递增的序列（队列尾为最小值），为了保存第二小的值的信息
        //队列中保存的是值的下标，为要保证滑动窗口的大小不超过k。（这里队列的大小并不是滑动窗口的大小！）
        //注意由于队列中保存的是值的下标，所以比较值的大小事，要记得转换；还有就是第一个元素的处理。
        vector<int> res;
        deque<int> dq;
        int len = nums.size();
        if(len == 1) return nums;
        dq.push_back(0);
        if(k == 1) res.push_back(nums[0]);
        for(int i = 1;i < len; i++){
            while(!dq.empty() && nums[i] > nums[dq.back()]){
                dq.pop_back();
            }
            dq.push_back(i);
            while(!dq.empty() && (i - dq.front() >= k)){
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

### 7.前K个高频元素

[347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

思路不难，`map`统计每个元素出现的次数，然后进行排序。要求时间复杂度为`O(n*logn)`，所以就要在排序上想办法。

看了自己第一次做这题的方法，是利用`sort`函数，重载比较函数，直接对`map`进行排序（应该是快排？），也是可以ac的（要注意比较函数的写法）。

题解中写的使用堆排序，可以直接使用`priority_queue`优先级队列，它的底层就是使用堆排序实现的。这里注意的就是`priority_queue`的初始化方法以及比较函数。

这里当然也可以自己手写堆排序，但是由于是对`map`排序，写起来会麻烦一些，可以考虑使用两个数组模拟`map`

```c
class Solution {
public:
    struct CmpByValue {  
        bool operator() (const pair<int,int>& lhs, const pair<int,int>& rhs) {  
            return lhs.second < rhs.second; //大顶堆
        }  
    };
    vector<int> topKFrequent(vector<int>& nums, int k) {
        map<int,int> cntMap;
        vector<int> res;
        for(int i = 0;i < nums.size(); i++){
            cntMap[nums[i]] ++;
        }
        //对map按照次数进行堆排序，返回前k大的元素
        priority_queue<pair<int,int>, vector<pair<int,int>>,CmpByValue> test;
        for(map<int,int>::iterator it = cntMap.begin(); it != cntMap.end(); it++){
            test.push(*it);
        }
        for(int i = 0;i < k; i++){
            res.push_back(test.top().first);
            test.pop();
        }
        return res;
    }
};
```



## 二叉树

### 1.二叉树的递归遍历

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

```c
//前序遍历
void preorderTraversal(TreeNode *root){
    if(root == nullptr) return;
    res.push_back(root->val);
    preorderTraversal(root->left);
    preorderTraversal(root->right);
}
//中序遍历
void inorderTraversal(TreeNode *root) {
    if(root == nullptr) return;
    inorderTraversal(root->left);
    res.push_back(root->val);
    inorderTraversal(root->right);
    
}
//后序遍历
void postorderTraversal(TreeNode *root){
    if(root == nullptr) return;
    postorderTraversal(root->left);
    postorderTraversal(root->right);
    res.push_back(root->val);
}
```



[589. N 叉树的前序遍历](https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/)

[590. N 叉树的后序遍历](https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/)

```c
class Solution {
public:
    vector<int> res;
    //后序
    void traverse(Node *root){
        if(root == NULL) return;
        for(int i = 0;i < root->children.size();i++){
            traverse(root->children[i]);
        }
        res.push_back(root->val);
    }
    //前序
    void traverse(Node *root){
        if(root == NULL) return;
        res.push_back(root->val);
        for(int i = 0;i < root->children.size();i++){
            traverse(root->children[i]);
        }
    }
    vector<int> preorder(Node* root) {
        traverse(root);
        return res;
    }
};
```

### 2.二叉树的迭代遍历*

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

前序遍历的迭代法的思想就是使用栈，

父节点先入栈，然后出栈。然后右孩子结点入栈，左孩子结点入栈（这样就是左孩子先出栈）。

再重复前面的出栈，右左孩子结点入栈，直到栈为空即可。

[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

迭代法的中序遍历的思路还是很容易想到的，就是要把最左侧的结点全部入栈，直到为空。

然后出栈，并把该节点的右节点入栈，重复上面的步骤（即最左侧结点全部入栈。）

这里的难点是【代码怎么实现】

首先重复的步骤是取某个点的全部最左侧结点，这需要一个循环来实现。

当这个结点为空时，就需要把栈顶元素出栈，取栈顶的元素的右子节点入栈。（遍历的过程中可能栈空，但是遍历没有结束）

[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

迭代法的后序遍历的思路，主要是在遍历左侧节点后，如果父节点存在右子节点，则应该先遍历右子节点，最后再出栈父节点。

但是再判断是否存在右子节点的过程中，如果当前是遍历完右子节点后返回的，则因为存在右子节点，则又会再次遍历右子节点，从而死循环。

所以在判断当前结点存在右子节点的时候，还要确定其不是从右子节点返回的，所以需要一个pre结点来记录上一个遍历的结点。即在出栈元素时要记录pre结点。

方法二是一种比较讨巧的思路。

由于先序遍历是中左右，而后序是左右中，所以我们只需要交换先序遍历为中右左，然后对结果取反即可。

```c
//迭代法-中序1
vector<int> inorderTraversal(TreeNode* root) {
    stack<TreeNode*> st;
    vector<int> res;
    if(root == NULL) return res;
    TreeNode *cur = root;
    while(!st.empty() || cur != nullptr){
        if(cur != nullptr) {
            st.push(cur);
            cur=cur->left;
        }else {
            TreeNode *node = st.top();
            st.pop();
            res.push_back(node->val);
            cur = node->right;
        }
    }
    return res;
}
//迭代法-中序2
vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> res;
        if(root == NULL) return res;
        TreeNode *cur = root;
        while(!st.empty() || cur != nullptr){
            while(cur != nullptr){
                st.push(cur);
                cur = cur->left;
            }
            cur = st.top();
            st.pop();
            res.push_back(cur->val);
            cur = cur ->right;
        }
        return res;
    }


//迭代法-前序
vector<int> preorderTraversal(TreeNode* root) {
    stack<TreeNode *> stk;
    vector<int> res;
    if(root == nullptr) return res;
    stk.push(root);
    while(!stk.empty() ){
        TreeNode *node = stk.top();
        res.push_back(node->val);
        stk.pop();
        if(node->right) stk.push(node->right);
        if(node->left) stk.push(node->left);
    }
    return res;
}

//迭代法-后序1
vector<int> postorderTraversal(TreeNode* root) {
    vector<int> res;
    stack<TreeNode *> st;
    TreeNode * cur = root;
    TreeNode *pre = nullptr;
    if (root == nullptr) return res;
    while(!st.empty() || cur != nullptr) {
        while(cur != nullptr) {
            st.push(cur);
            cur = cur->left;
        }
        cur = st.top();
        if(cur -> right && cur->right != pre) 
            cur = cur->right;
        else {
            st.pop();
            res.push_back(cur->val);
            pre = cur;
            cur = nullptr;
        }
    }
    return res;
}

//迭代法-后序2
vector<int> postorderTraversal(TreeNode* root) {
    vector<int> res;
    stack<TreeNode *> st;
    if(root == nullptr) return res;
    st.push(root);
    while(!st.empty()){
        TreeNode *node =st.top();
        st.pop();
        res.push_back(node->val);
        if(node->left) st.push(node->left);
        if(node->right) st.push(node->right);
    }
    reverse(res.begin(), res.end());
    return res;
}
```



### 3.二叉树的层次遍历

[102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```c
vector<vector<int>> levelOrder(TreeNode* root) {
    queue<TreeNode*> qe;
    vector<vector<int>> res;
    if(root == nullptr) return res;
    res.push_back({root->val});
    qe.push(root);
    while(!qe.empty()){
        int len = qe.size();
        vector<int> temp;
        while(len--){
            TreeNode *node = qe.front();
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
```

[107. 二叉树的层序遍历 II](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)

反转层次遍历的结果即可	

```c
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        queue<TreeNode*> qe;
        vector<vector<int>> res;
        if(root == nullptr) return res;
        qe.push(root);
        res.push_back({root->val});
        while(!qe.empty()){
            vector<int> temp;
            int len = qe.size();
            while(len--){
                TreeNode *node= qe.front();
                qe.pop();
                if(node->left) {
                    qe.push(node->left);
                    temp.push_back(node->left->val);
                }
                if(node->right){
                    qe.push(node->right);
                     temp.push_back(node->right->val);
                }
            }
            if(temp.size() !=0) res.push_back(temp);
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
```



[199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

其实就是求每一层的最右结点

在层次遍历的过程中所有的结点都要保存到队列中，因为不确定最右结点会出现在哪个结点下面。但是只把最右结点（每一层的最后一个元素的左或者右结点）保存至res数组即可。

也可以使用flag标记当前层是否以及存入元素到res数组。

```c
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        queue<TreeNode*> qu;
        vector<int> res;
        if(root == nullptr) return res;
        qu.push(root);
        res.push_back(root->val);
        while(!qu.empty()){
            int len = qu.size();
            int flag = 0;
            while(len --){
                TreeNode *node = qu.front();
                qu.pop();
                if(node->right){
                    qu.push(node->right);
                    if(flag == 0) {
                        res.push_back(node->right->val);
                        flag = 1;
                    }
                }
                if(node->left) {
                    qu.push(node->left);
                    if(flag == 0) {
                        res.push_back(node->left->val);
                        flag = 1;
                    }
                }
            }
        }
        return res;
    }
};
```



[637. 二叉树的层平均值](https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/)

```c
class Solution {
public:
    vector<double> averageOfLevels(TreeNode* root) {
        queue<TreeNode*> qu;
        vector<double> res;
        if(root == nullptr) return res;
        qu.push(root);
        res.push_back(root->val);
        while(!qu.empty()){
            int len = qu.size();
            double sum = 0;
            int num = 0;
            while(len --){
                TreeNode *node = qu.front();
                qu.pop();
                if(node->right){
                    qu.push(node->right);
                    sum += node->right->val;
                    num++;
                }
                if(node->left) {
                    qu.push(node->left);
                    sum+=node->left->val;
                    num++;
                }
            }
            if(num != 0) res.push_back((sum*1.0)/num);
        }
        return res;
    }
};
```



[515. 在每个树行中找最大值](https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/)

层次遍历即可

```c
class Solution {
public:
    vector<int> largestValues(TreeNode* root) {
        vector<int> res;
        queue<TreeNode*> qu;
        if(root == nullptr) return res;
        qu.push(root);

        while(!qu.empty()){
            int len = qu.size();
            int maxNode = INT_MIN;
            while(len --){
                TreeNode *node=qu.front();
                qu.pop();
                maxNode = max(maxNode, node->val);
                if(node->left) {
                    qu.push(node->left);
                }
                if(node->right) {
                    qu.push(node->right);
                }
                
            }
            res.push_back(maxNode);
        }
        return res;
    }
};
```



[429. N 叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)

层次遍历即可

```c
class Solution {
public:
    vector<vector<int>> levelOrder(Node* root) {
        vector<vector<int>> res;
        queue<Node*> qu;
        if (root == nullptr) return res;
        qu.push(root);
        
        while(!qu.empty()){
            int len = qu.size();
            vector<int> temp;
            while(len --) {
                Node *node = qu.front();
                qu.pop();
                temp.push_back(node->val);
                int l = node->children.size();
                for(int i = 0;i < l;i ++){
                    if(node->children[i] != nullptr) {
                        qu.push(node->children[i]);
                    }
                }
            }
            if(!temp.empty()) res.push_back(temp);
        }
        return res;
    }
};
```



[116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

队列的层次遍历：`O(n)`空间复杂度

思路比较简单，只需要对当前队列里面的所有节点做处理即可。

`O(1)`空间复杂度的方法就不能使用队列，有递归法和遍历法。主要是需要直接对每个节点做处理。

**递归法：**对于每一个节点，处理其子节点。

- 如果左子节点不为空，就指向右子节点
- 如果当前节点的`next`的节点不为空，则要把当前节点的右子结点指向`next`节点所指向节点的左子节点
- 递归左右子节点

**遍历法：**

根据递归的思想，把其写成遍历的方式。要注意的时该树是完美二叉树。

每次只要遍历其最`left`节点即可。

```c
//空间复杂度是O(n)
class Solution {
public:
    Node* connect(Node* root) {
        Node * res;
        if(root == NULL) return res;
        root->next == NULL;
        queue<Node*> qu;
        qu.push(root);
        while(!qu.empty()){
            int len = qu.size();
            while(len--){
                Node *node = qu.front();
                qu.pop();
                Node *nextNode = NULL;
                if(len) nextNode = qu.front(); //注意这里是if(len)
                node->next = nextNode;
                if(node->left){
                    qu.push(node->left);
                }
                if(node->right){
                    qu.push(node->right);
                }
            }
        }
        return root;
    }
};

//递归法
class Solution {
public:
    Node* connect(Node* root) {
        if(root == NULL) return NULL;
        if(root->left) root->left->next = root->right;
        if(root->next && root->right) {
            root->right->next = root->next->left;
        }
        connect(root->left);
        connect(root->right);
        return root;
    }
};

//迭代法
class Solution {
public:
    Node* connect(Node* root) {
        if(root == NULL) return root;
        Node *pre =  root;
        Node *cur = NULL;
        while(pre->left){ //prefect binary tree
            cur = pre;
            while(cur){
                cur->left->next = cur->right;
                if(cur->next){
                    cur->right->next = cur->next->left;
                }
                cur = cur->next;
            }
            pre = pre->left; //next level
        }
        return root;
    }
};
```

[117. 填充每个节点的下一个右侧节点指针 II](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/)

这题与上面116这题的区别就在于这题的树并不是一个完全二叉树。

如果使用队列进行遍历求解，则并没有什么需要修改的点。

如果使用`O(1)`空间复杂度的递归法进行求解，则有一些需要注意的点。

- `root`的节点的右子结点的`next`指针，可能需要指向`root`的`next`串上的任意一个节点
- 在递归的过程中，需要变递归遍历右子结点，再遍历左子节点。
    - 因为对`root`的左子节点进行处理时，需要`root`的`next`全部建立完成。
    - 而非完全二叉树递归的过程中，如果先遍历左子节点，则右边的`next`串并没有完全建立

O(1)空间的迭代版也是对层次遍历的优化

- 在遍历的过程中，使用一个指针记录下一层的关系，并且使用一个指针遍历每一层的所有结点

```c
//队列遍历
class Solution {
public:
    Node* connect(Node* root) {
        queue<Node*> qu;
        if(root == NULL) return root;
        qu.push(root);
        while(!qu.empty()){
            int len = qu.size();
            while(len--){
                Node *node = qu.front();
                qu.pop();
                Node *nextNode = NULL;
                if(len) nextNode = qu.front();
                node->next = nextNode;
                if(node->left) qu.push(node->left);
                if(node->right) qu.push(node->right);
            }
        }
        return root;
    }
    
};

// 递归法
class Solution {
public:
    Node* connect(Node* root) {
        //递归
        if(root == NULL) return NULL;
        if(root->left){
            if(root->right) {
                root->left->next = root->right;
            }
            else if(root->next){
                Node *temp = root->next;
                while(temp!=NULL){ //要找到该节点的next的节点，可能出现再root节点的next串上的任何一个节点中
                    if(temp->left){
                        root->left->next = temp->left;
                        temp = NULL;
                    }else if(temp->right){
                        root->left->next = temp->right;
                        temp = NULL;
                    }else if (temp->next){
                        temp = temp->next;
                    }else {
                        temp = NULL;
                    }
                }
            }
        }
        if(root->right){
            if(root->next){
                Node *temp = root->next;
                while(temp!=NULL){
                    if(temp->left){
                        root->right->next = temp->left;
                        temp = NULL;
                    }else if(temp->right){
                        root->right->next = temp->right;
                        temp = NULL;
                    }else if (temp->next){
                        temp = temp->next;
                    }else {
                        temp = NULL;
                    }
                }
            }
        }
        connect(root->right); //这要先遍历右子结点，再遍历左子节点
        connect(root->left);
        return root;

        //        1
        //    2       3
        //  4   5   x   6    如果先遍历左结点，再遍历右结点，在遍历4的时候，5 ，6 之间的next关系还没有被建立
        // 7 x x x x 8 x x
    }
};

//O(1)的迭代版
class Solution {
public:
    Node* connect(Node* root) {
        if(root == NULL) return root;
        Node *cur = root;
        while(cur != NULL){
            Node *head = new Node(0);
            Node *pre = head;
            while(cur != NULL){
                if(cur->left){
                    head->next = cur->left;
                    head = head->next;
                }
                if(cur->right){
                    head->next = cur->right;
                    head = head->next;
                }
                cur = cur->next;
            }
            cur = pre->next;
        }
        return root;
    }
};
```



[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

【递归法】

- depth函数计算当前结点的深度
- 对于某一个结点而言，该节点的深度等于其左子节点深度与右子结点深度的最大值+1

【层次遍历法】

- 直接记录层数

```c
//递归
class Solution {
public:
    int maxDepth(TreeNode* root) {
        return depth(root);
    }
    int depth(TreeNode *root){
        if(root == nullptr) {
            return 0;
        }
        int left = depth(root->left);
        int right = depth(root->right);
        return max(left,right) + 1;
    }
};

//层次遍历->queue
class Solution {
public:
    int maxDepth(TreeNode* root) {
        queue<TreeNode*> qu;
        if(root == nullptr) return 0;
        qu.push(root);
        int depth = 0;
        while(!qu.empty()){
            int len = qu.size();
            while(len--){
                TreeNode *node = qu.front();
                qu.pop();
                if(node->left) qu.push(node->left);
                if(node->right) qu.push(node->right);
            }
            depth++;
        }
        return depth;
    }
};
```

[111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

【递归法】

- 叶子结点时，深度为1
- 如果左右子结点中，有一个子节点为null，另一个不为null时，则不应该最直接返回`min(left,depth)+1`,因为会把空子节点的深度返回，但是这并不是所需要的深度。

【层次遍历法】

- 遇到叶子结点就返回深度即可

```c
class Solution {
public:
    int minDepth(TreeNode* root) {
        if(root == nullptr) return 0;
        return depth(root);
    }
    int depth(TreeNode* root){
        if(root == nullptr) return 0;
        if(root->left == nullptr && root->right == nullptr) return 1;  //叶子结点时，返回的深度为1
        if(root->left == nullptr && root->right) return 1+depth(root->right);
        if(root->left && root->right == nullptr) return 1+depth(root->left);
        int left = depth(root->left);
        int right = depth(root->right);
        return min(left,right)+1;
    }
};


class Solution {
public:
    int minDepth(TreeNode* root) {
        if(root == nullptr) return 0;
        queue<TreeNode*> qu;
        qu.push(root);
        int depth = 0;
        while(!qu.empty()){
            int len = qu.size();
            depth++;
            while(len--){
                TreeNode *node = qu.front();
                qu.pop();
                if(node->left == nullptr && node->right == nullptr) return depth;
                if(node->left) qu.push(node->left);
                if(node->right) qu.push(node->right);
            }
        }
        return depth;
    }
};
```



### 4.翻转二叉树

[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

```c
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(root == nullptr) return root;
        reverse(root);
        return root;
    }
	//先序
    void reverse(TreeNode *root){
        if(root == nullptr) return;
        swap(root->left,root->right);
        reverse(root->left);
        reverse(root->right);
    }
    //中序
    void reverse(TreeNode *root){
        if(root == nullptr) return;
        reverse(root->left);
        swap(root->left,root->right);
        reverse(root->left); //因为已经交换了左右结点，所以这里递归左结点
    }
    //后序
    void reverse(TreeNode *root){
        if(root == nullptr) return;
        reverse(root->left);
        reverse(root->right);
        swap(root->left,root->right);
    }
};

//层次遍历，交换每个节点的左右子节点
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(root == nullptr) return root;
        queue<TreeNode*> qu;
        qu.push(root);
        while(!qu.empty()){
            int len = qu.size();
            while(len--){
                TreeNode *node = qu.front();
                qu.pop();
                swap(node->left,node->right);
                if(node->left) qu.push(node->left);
                if(node->right) qu.push(node->right);
            }
        }
        return root;
    }
};
```



### 5.对称二叉树

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

【递归法】

- 对于当前节点，要判断其左子结点和右子结点是否对程（包括值）
- 如果对称，就要递归比较子结点的子树是否对称。
    - 这里左子节点的左子节点应与右子结点的右子结点进行比较。
    - 这里左子节点的右子节点应与右子结点的左子结点进行比较。
    - 只有当上面的两个都对称时，才对称。

【遍历法】

- 使用层次遍历，对每层的序列进行是否对称的判断。

```c
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(root == nullptr) return true;
        return func(root->left, root->right);
    }
    bool func(TreeNode *left, TreeNode *right){
        if(!left && !right) return true;
        else if(!left && right) return false;
        else if(left && !right) return false;
        else if(left->val == right->val) return func(left->left, right->right) && func(left->right, right->left);
        else return false;
    }
};
```

### 6.二叉树的最大深度

[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

见【二叉树的层次遍历】



[559. N 叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-n-ary-tree/)

【递归法】

- 值得注意的-见注释

【遍历法】

- 层次遍历记录深度即可

```c
class Solution {
public:
    int maxDepth(Node* root) {
        if(root == NULL) return 0;
        int d = 0; //这里必须d为0，当root为叶子节点时，返回的深度应该为1 = d +1
        for(int i = 0;i < root->children.size();i++){
            d =  max(maxDepth(root->children[i]),d);
        }
        return d + 1;
    }
};
```

### 7.二叉树的最小深度

[111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

见【二叉树的层次遍历】----主要是递归法



### 8.完全二叉树的节点个数

[222. 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/)

【首先对于一个普通的二叉树而言】

迭代法：使用层次遍历即可

递归法：对于每个结点：为空则返回0；不为空则返回子节点的个数+1

```c
class Solution {
public:
    int countNodes(TreeNode* root) {
        if(root == nullptr) return 0;
        else return Num(root);
    }
    int Num(TreeNode *node){
        if(node == nullptr) return 0;
        return Num(node->left) + Num(node->right) + 1;
    }
};
```

【对于一个完全二叉树而言】

在递归的过程中，对于某一个节点而言，要统计该节点为root的子树的全部节点数

- 该子树的节点数为：左右子树的结点个数之和
- 这里可以处理的为：
    - 当该子树为满二叉树时，就不需要遍历全部结点。对于满二叉树，其节点的个数等于（层数的平方-1）
        - 又该树为完全二叉树，所以对于完全二叉树而言，如果最左结点和最右结点在同一层时，那么它就是一个满二叉树。
        - 所以只要计算最左和最右结点的深度即可
    - 当子树不为满二叉树时，就只需要返回左右子树的节点数的和+1

```c
class Solution {
public:
    int countNodes(TreeNode* root) {
        if(root == nullptr) return 0;
        TreeNode *leftnode = root->left;
        TreeNode *rightnode = root->right;
        int leftdepth = 0;
        int rightdepth = 0;
        while(leftnode){
            leftnode = leftnode->left;
            leftdepth++;
        }
        while(rightnode){
            rightnode = rightnode->right;
            rightdepth++;
        }
        if(leftdepth == rightdepth) return (2 << leftdepth) - 1;
        return countNodes(root->left) + countNodes(root->right) + 1;
    }
};
```



### 9.平衡二叉树

[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

这里要想清楚的递归函数的作用的是什么。

如果递归函数直接返回对平衡二叉树判断的结果，似乎是不好写的，因为这需要子树的深度，但是对于某一个节点而言，其子树的深度是不知道的。

所以我们这里是用递归函数求子树的深度，对递归过程中的左右子树的深度做判断，对单独的变量对赋值。

```c
class Solution {
public:
    bool res = true;
    bool isBalanced(TreeNode* root) {
        if(root == nullptr) return true;
        traverse(root);
        return res;
    }
    int traverse(TreeNode *root){
        if(root == nullptr) return 0;
        int l = traverse(root->left) + 1;
        int r = traverse(root->right) + 1;
        if(abs(l-r) > 1) res =false;
        return l>r?l:r;  //子树的深度为左右子节点的最深的深度
    }
};
```



### 10.二叉树的所有路径

[257. 二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/)

【递归法】

首先容易想到的就是递归法，这里对于每个结点而言，分情况讨论：

- 如果是null，直接返回
- 对于非空结点，都要把该点拼接到字符串上
- 如果是叶子节点，则在拼接字符串之后把字符串存入结果中
- 如果是其他结点，则遍历其左右子树，并在字符串后面加上"->"

这里注意的要记录从根到叶子的路径，所以要保存从上到下的经过的每个节点，而不是从叶子节点向上返回。

所以在传递的过程中需要把字符串作为参数。

【迭代法】

迭代前序遍历的过程中，保存路径即可。

问题是遍历的过程中路径如何存储？因为路径的个数是无法确定的。

所以在遍历的过程中，当结点出栈时，我们怎么保存字符串？

解决方案是再使用一个栈存储遍历的路径（每到一个结点会新生成一道路径string），这样每次出栈的路径就是到达当前结点的路径。

```c
class Solution {
public:
    vector<string> res;
    vector<string> binaryTreePaths(TreeNode* root) {
        if(root == nullptr) return res;
        string temp="";
        traverse(root, temp);
        return res;
    }
    void traverse(TreeNode *root, string temp){
        if(root == nullptr) return;
        string s = to_string(root->val);
        temp += s;
        if(!root->left && !root->right){
            res.push_back(temp);
        }
        traverse(root->left, temp+"->");
        traverse(root->right,temp+"->");
    }
};

// 迭代
class Solution {
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> res;
        if(root == nullptr) return res;
        stack<TreeNode*> st;
        st.push(root);
        stack<string> path;
        path.push(to_string(root->val));
        while(!st.empty()){
            TreeNode *node = st.top();
            st.pop();
            string p = path.top();
            path.pop();
            if(!node->left && !node->right){
                res.push_back(p);  
            }
            if(node->right){
                st.push(node->right);
                path.push(p+"->"+to_string(node->right->val));
            }
            if(node->left){
                st.push(node->left);
                path.push(p+"->"+to_string(node->left->val));
            }
        }
        return res;
    }
};
```





### 11.相同的树

[100. 相同的树](https://leetcode-cn.com/problems/same-tree/)

```c
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(!p && !q) return true;
        else if(!p || !q) return false;
        else if(p->val != q->val) return false;
        else return isSameTree(p->left,q->left) && isSameTree(p->right,q->right);
    }
};
```



### 12.另一棵树的子树

[572. 另一棵树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/)

思路很容易想到，就是对于root中的每一个结点进行和subroot是否是同一棵树的判断

相同的树的判断以及在100题中写过，在主函数中对于root结点，及其左右子节点进行判断。

要注意的是，在主函数中，`return isSubtree(root->left,subRoot) || isSubtree(root->right,subRoot)`中递归调用的是主函数，而不是`dfs`函数!

```c
class Solution {
public:
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        if(root == nullptr) return false;
        if(dfs(root,subRoot)) return true;
        return isSubtree(root->left,subRoot) || isSubtree(root->right,subRoot);
    }
    bool dfs(TreeNode *p, TreeNode *q){
        if(!p && !q) return true;
        else if(!p || !q) return false;
        else if(p->val != q->val) return false;
        else return (dfs(p->left,q->left) && dfs(p->right,q->right));
    }
};
```





### 13.左叶子之和

[404. 左叶子之和](https://leetcode-cn.com/problems/sum-of-left-leaves/)

【递归法】

在递归遍历的过程中，对左叶子进行判断

【迭代法】

只需要在迭代的过程中对左叶子进行判断即可

```c
class Solution {
public:
    int res = 0;
    int sumOfLeftLeaves(TreeNode* root) {
        func(root);
        return res;
    }
    void func(TreeNode *root){
        if(!root) return;
        if(root->left)
        if(root->left){
            if(!root->left->left && !root->left->right )
                res += root->left->val;
            func(root->left);
        }
        if(root->right) func(root->right);
    }
};
```



### 14.找树左下角的值

[513. 找树左下角的值](https://leetcode-cn.com/problems/find-bottom-left-tree-value/)

【v1】

是第一版代码，思路上是正确的，通过深度遍历，在遍历的过程中记录当前结点的深度，对于每一层的结点，记录最左结点。

还需要记录当前的最深的结点以及是否已经记录最左结点，使用了depth保存了当前最深的最左结点的层数，使用flag标识当前层是否已经保存结点值。

由于使用先序遍历，所以每层第一个遍历的就是最左结点。在下一层时（d+1 > depth），把flag置为0



【v2】

参考了【代码随想录】的答案，就不要flag来标记当前层是是否已经保存。

如果对【v1】的代码进行修改，去掉flag也是正确的，因为每层第一个遍历的就是最左结点，而该层的其他结点的深度不会大于最左结点的深度。



```c
// v1
class Solution {
public:
    int depth = 0;
    int res = 0;
    int flag = 0;
    int findBottomLeftValue(TreeNode* root) {
        if(root && !root->left &&!root->right) return root->val;
        dfs(root,0);
        return res;
    }
    void dfs(TreeNode *root, int d){
        if(d+1 > depth) flag = 0;
        if(root == nullptr) return;
        if(root->left){
            if(root->left->left == nullptr && root->left->right == nullptr) {
                if(d+1 > depth && flag == 0){
                    res = root->left->val;
                    flag = 1;
                    depth = d+1;
                }
            }
        }
        if(root->right){
            if(root->right->left == nullptr && root->right->right == nullptr) {
                if(d+1 > depth && flag == 0){
                    res = root->right->val;
                    flag = 1;
                    depth = d+1;
                }
            }
        }
        dfs(root->left, d+1);
        dfs(root->right,d+1);
    }
};

// v2
class Solution {
public:
    int depth = INT_MIN;
    int res = 0;
    int findBottomLeftValue(TreeNode* root) {
        if(root && !root->left &&!root->right) return root->val;
        dfs(root,0);
        return res;
    }
    void dfs(TreeNode *root, int d){
        if(!root->left && !root->right){
            if(d > depth){
                res = root->val;
                depth = d;
            }
        }
        if(root->left){
            d++;
            dfs(root->left,d);
            d--;
        }
        if(root->right){
            d++;
            dfs(root->right,d);
            d--;
        }
    }
};
```



### 15.路径总和*

[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

【递归法】

递归只需要在遍历的过程中计算当前距离targetsum的差

当遇到叶子节点时，如果差为0，则表示存在

【迭代法】

如果使用迭代法，就要记录当遍历到叶子结点的和，因为要进行回溯，类似[257. 二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/)的迭代法一样

所以如何记录当前结点的总和，再利用栈存储一个这样的数据

```c
// 递归
class Solution {
public:
    bool res = false;
    bool hasPathSum(TreeNode* root, int targetSum) {
        if(root == nullptr) return false;
        traverse(root,targetSum);
        return res;
    }
    void traverse(TreeNode *root, int sum){
        if(root == nullptr) return;
        if(root->left == nullptr && root->right == nullptr){
            sum -= root->val;
            if(sum == 0){
                res = true;
            }
        }
        if(root->left){
            traverse(root->left,sum-root->val);
        }
        if(root->right){
            traverse(root->right,sum-root->val);
        }
    }
};

// 迭代
class Solution {
public:
    bool res = false;
    bool hasPathSum(TreeNode* root, int targetSum) {
        if(root == nullptr) return false;
        stack<TreeNode*> stnode;
        stack<int> stsum;
        stnode.push(root);
        stsum.push(targetSum-root->val);
        while(!stnode.empty()){
            TreeNode *node = stnode.top();
            stnode.pop();
            int tempsum = stsum.top();
            stsum.pop();
            if(node->left==nullptr && node->right==nullptr){
                if(tempsum == 0) res = true;
            }
            if(node->right) {
                stnode.push(node->right);
                stsum.push(tempsum - node->right->val);
            }
            if(node->left){
                stnode.push(node->left);
                stsum.push(tempsum - node->left->val);
            } 
        }
        return res;
    }
};
```



### 16.从遍历序列构造二叉树

[106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

由后序序列的性质可知，一棵（子）树的后序序列的最后一个结点即为根节点。

这个性质可以方便构造二叉树，因为我们首先就获得了根节点。

其次，对于该根节点，我们需要构造其左子树与右子树，

而由中序遍历的过程可知，序列中根节点的左边全部否是左子树结点，而右边全部都是右子树结点。

所以我们可以通过后序序列获得根节点，通过中序序列获得左右子树的结点。

然后我们只要递归遍历子树序列就够可以构造出该二叉树了。

由于序列是在不断的变化的，所以我们需要指定中序序列以及后序序列的左右端点（inl，inr，postl, postr）。

假设在树的中序序列中我们找到树的根节点坐标为i,那么 （inl,i-1） 即为左子树的序列，（i+1,inr）即为右子树的序列

那么左子树和右子树的根节点又要怎么确定呢？我们一定需要后序序列的性质。

由后序序列可知，如果树的后序序列中，根结点坐标为j,那么，右子树的根节点一定为j-1（第一个为postl+left），

重点是左子树的根节点要怎么求呢？？

由于我们以及知道了中序序列中，根节点左边都是左子树结点，那么我们就可以求出左子树的节点个数为left

那么在后序序列中，左子树的根节点就存在于postl+left-1的位置上，即左子树序列的最后一个（第一个为postl）

所以对于左右子树而言，中序和后序序列的左右端点都是需要的！！！

最后，当inr > inl时，说明没有子树结点了，返回空即可。

```c
class Solution {
public:
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
    return traverse(inorder,0,inorder.size()-1,postorder, 0, postorder.size()-1);
    }

    TreeNode* traverse(vector<int> &inorder, int inl, int inr, vector<int>& postorder, int postl,int postr){
        if(inl > inr) return NULL;
        TreeNode *node = new TreeNode(postorder[postr]);
        int index = inl;
        for(index = inl; index <= inr; index++) {
            if(inorder[index] == postorder[postr]) break;
        }
        int left = index - inl;
        node->left = traverse(inorder, inl, index-1, postorder, postl, postl+left-1);
        node->right = traverse(inorder, index+1 ,inr, postorder, postl+left, postr - 1);
        return node;
    }
};
```



[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

思路与上一题是一样的，不同的就是左右端点的处理。

```c
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return traverse(preorder, 0, preorder.size() - 1, inorder, 0 ,inorder.size() - 1);
    }
    TreeNode *traverse(vector<int> &preorder, int prel, int prer, vector<int>& inorder, int inl, int inr){
        if(inl > inr) return nullptr;
        TreeNode *node = new TreeNode(preorder[prel]);
        int index = inl;
        for(;index <= prer; index++){
            if(inorder[index] == preorder[prel]) break;
        }
        int left = index - inl;
        node->left = traverse(preorder,prel + 1, prer+left, inorder, inl, index - 1);
        node->right = traverse(preorder, prel + left + 1,prer, inorder, index+1, inr);
        return node;
    }
};
```

### 17.最大二叉树

[654. 最大二叉树](https://leetcode-cn.com/problems/maximum-binary-tree/)

递归构造二叉树即可

```c
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return traverse(nums,0,nums.size()-1);
    }

    TreeNode* traverse(vector<int> nums, int l, int r){
        if(l > r) return nullptr;
        int index = l;
        int max = INT_MIN;
        for(int i = l; i <= r; i ++){
            if(nums[i] > max){
                max = nums[i];
                index = i;
            } 
        }
        TreeNode *node = new TreeNode(max);
        node->left=traverse(nums,l,index-1);
        node->right=traverse(nums,index+1,r);
        return node;
    }
};
```

### 18.合并二叉树

[617. 合并二叉树](https://leetcode-cn.com/problems/merge-two-binary-trees/)

```c
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        if(!root1 && !root2) return root1;
        if(!root1 && root2) return root2;
        if(root1 && !root2) return root1;
        root1->val += root2->val;
        root1->left = mergeTrees(root1->left, root2->left);
        root1->right = mergeTrees(root1->right,root2->right);
        return root1;
    }
};
```

### 19.二叉搜索树中的搜索

[700. 二叉搜索树中的搜索](https://leetcode-cn.com/problems/search-in-a-binary-search-tree/)

```c
class Solution {
public:
    TreeNode *res = nullptr;
    TreeNode* searchBST(TreeNode* root, int val) {
        traverse(root,val);
        return res;
    }
    void traverse(TreeNode *root, int val){
        if (root == nullptr) return;
        if (root->val == val) {
            res = root;
        }
        searchBST(root->left,val);
        searchBST(root->right,val);
    }
};
```

### 20.验证二叉搜索树

[98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

二叉搜索树的最重要的性质就是其中序序列是一个递增序列，所以可以对中序序列进行判断

或者直接在二叉树的遍历过程中进行判断（需要记录每个结点的值的区间）

```c
// 利用二叉搜索树的中序序列是一个递增序列的性质
class Solution {
public:
    bool res = true;
    vector<int> v;
    bool isValidBST(TreeNode* root) {
        traverse(root);
        return judge();
    }
    void traverse(TreeNode *root){
        if(!root) return;
        traverse(root->left);
        v.push_back(root->val);
        traverse(root->right);
    }
    bool judge(){
        for(int i = 1;i < v.size();i ++){
            if(v[i-1] >= v[i]){
                return false;
            }
        }
        return true;
    }
};

//按照二叉搜索树的定义进行递归
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return traverse(root, INT64_MIN, INT64_MAX);
    }
    bool traverse(TreeNode *root, long int minnode, long int maxnode){
        if(root == nullptr) return true;
        if(root->val >= maxnode || root->val <= minnode) return false;
        return traverse(root->left, minnode, root->val) && traverse(root->right, root->val, maxnode);
    }
};
```



### 21.二叉搜索树的最小绝对差

[530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)

因为中序序列是一个递增序列，只需要在中序遍历过程中，记录前一个结点和当前结点的差值的最小值即可。

```c
class Solution {
public:

    int res = INT_MAX;
    TreeNode *pre;
    int getMinimumDifference(TreeNode* root) {
        traverse(root);
        return res;
    }
    void traverse(TreeNode *root) {
        if(root == nullptr) return; 
        traverse(root->left);
        if(pre !=nullptr)
            res = min(res, abs(pre->val - root->val));
        pre  = root;
        traverse(root->right);
    }
};
```

### 22.二叉搜索树中的众数

[501. 二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)

对于任意的二叉树而言，我们只需要统计二叉树的元素出现的次数即可

```c
// 任意二叉树
class Solution {
public:
    vector<int> result;
    map<int,int> cntmap;
    bool static cmp(const pair<int,int>& a, const pair<int,int> &b){
        return a.second > b.second;
    }
    vector<int> findMode(TreeNode* root) {
        traverse(root);
        vector<pair<int, int>> vec(cntmap.begin(), cntmap.end());
        sort(vec.begin(), vec.end(), cmp);
        result.push_back(vec[0].first);
        for (int i = 1; i < vec.size(); i++) {
            if (vec[i].second == vec[0].second) result.push_back(vec[i].first);
            else break;
        }
        return result;
    }
    void traverse(TreeNode *root){
        if(root==nullptr) return;
        cntmap[root->val]++;
        traverse(root->left);
        traverse(root->right);
    }
};
```

### 23.二叉树的最近公共祖先

[236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

【递归法】

完全使用递归，对于某一层，

- 如果当前结点等于p或者q，或者为null则返回该结点
- 如果都是所要找的结点，则需要寻找当前结点是否是最近公共祖先，所以看左子树或者右子树中是否存在p或者q节点
    - 如果左右子树返回的都不为null，则说明root为最近祖先（最先返回的root的会一直被返回）
    - 某一个为空，说明另一个返回的就为最近祖先

【常规】

用栈记录从root到p 以及 root到q的路径

出栈到栈的大小一致后，共同出栈，找到公共祖先即可

【注意】这里有一个递归遍历单边的方式

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == p || root == q || root == NULL) return root;
        TreeNode *left = lowestCommonAncestor(root->left, p , q);
        TreeNode *right = lowestCommonAncestor(root->right, p , q);
        if(left && right) return root;
        else if(left && !right) return left;
        else if(!left && right) return right;
        return NULL;
    }

};

//stack
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        stack<TreeNode *> m1;
        stack<TreeNode *> m2;
        getpath(root,p,m1);
        getpath(root,q,m2);
        while(m1.size() != m2.size())
        {
            if(m1.size() > m2.size())
                m1.pop();
            else
                m2.pop();
        }
        while(m1.top() != m2.top())
        {
            m1.pop();
            m2.pop();
        }
        return m1.top();
    }

    bool getpath(TreeNode* root,TreeNode* node,stack<TreeNode*>& path)
    {
        if(root == NULL)
            return false;
        path.push(root);
        if(root == node)
            return true;
        if(getpath(root->left,node,path))  // 递归遍历单边，使用bool返回值
            return true;
        if(getpath(root->right,node,path))
            return true;
        path.pop();
        return false;       
    }
};

```

### 24.二叉搜索树中的插入操作

[701. 二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

这里也是用单边遍历的方式，如果在某一侧插入，则另一边就不需要再遍历。

插入的结点一定是在叶子结点或者是某一侧是空的结点上。

然后再遍历的过程中进行判断即可。

```c++
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if(root == nullptr) {
            TreeNode *node = new TreeNode(val);
            return node;
        }
        traverse(root,val);
        return root;
    }
    bool traverse(TreeNode *root, int val) {
        if(root == nullptr) return false;
        if(root->left == nullptr || root->right == nullptr) {
            TreeNode *node = new TreeNode(val);
            if(val < root->val && root->left == nullptr){
                root->left = node;
                return true;
            } 
            if(val >= root->val && root->right == nullptr) {
                root->right = node;
                return true;
            }
            
        }
        if(root->val > val) {
            if(traverse(root->left,val)) return true;
        }
        if(root->val <= val) {
            if(traverse(root->right,val)) return true;
        }
        return false;
    }
        //         8
        //     x     55
        //        39    x
        //     11   x
        //   x    23
        //       x  x  

    //     5
    // x       14
    //      10    77
    //     x  x  x  95 
};
```

### 25.删除二叉搜索树中的节点

[450. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)

这里由于设计二叉搜索树的调正，所以我的想法是，把key结点删除之后，把key结点的子节点通过遍历的方式插入子树中。

在上面已经写过二叉搜索树的插入，这里还要注意的就是删除根节点的情况。

如果是删除根节点，则把左子树插入右子树中，或者把右子树插入左子树中即可。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode *rootNode;
    TreeNode* deleteNode(TreeNode* root, int key) {
        if(root == nullptr) return root;
        if(root->val == key) {
            if(root->right == nullptr) {
                rootNode = root->left;
                subTree(root->right);
                return root->left;
            }else{
                rootNode = root->right;
                subTree(root->left);
                return root->right;
            }
        }
        rootNode = root;
        adjust2(root,key,nullptr);
        return root;

    }
    bool adjust2(TreeNode *root, int key, TreeNode *pre){
        if(root == nullptr) return false;
        if(root->val == key) {
            if(pre->left == root) {
                pre->left = nullptr;
            }else {
                pre->right = nullptr;
            } 
            subTree(root->left);
            subTree(root->right);
        }
        if(adjust2(root->left,key,root)) return true;
        if(adjust2(root->right, key, root)) return true;
        return false;
    }
    
    void subTree(TreeNode* node) {
        if(node == nullptr) return;
        TreeNode *t = new TreeNode(node->val);
        insert(rootNode, t);
        subTree(node->left);
        subTree(node->right);   
    }

    bool insert(TreeNode *root, TreeNode *node) {
        if(root == nullptr) return false;
        if(root->left == nullptr || root->right == nullptr) {
            if(node->val < root->val && root->left == nullptr){
                root->left = node;
                return true;
            } 
            if(node->val >= root->val && root->right == nullptr) {
                root->right = node;
                return true;
            }
            
        }
        if(root->val > node->val) {
            if(insert(root->left,node)) return true;
        }
        if(root->val <= node->val) {
            if(insert(root->right,node)) return true;
        }
        return false;
    }
};
```

### 26.修剪二叉树

[669. 修剪二叉搜索树](https://leetcode-cn.com/problems/trim-a-binary-search-tree/)

对于[low, high]区间之外的点要删除，本来删除之后的结点的子树又涉及树的调整

但是这里是搜索树，所以

如果是小于low的结点，删除之后，该节点的左子树都小于low，则只需要对其右子树进行处理

如果是大于high的结点，删除之后，该结点的右子树都大于high，则只需要对其左子树进行处理

如果都在区间之内，则对左右子树进行处理

然后返回处理后的结点

```c++
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        if(root == nullptr) return root;
        if(root->val < low) {
            TreeNode *right = trimBST(root->right , low ,high);
            return right;
        }
        if(root->val > high) {
            TreeNode *left = trimBST(root->left , low, high);
            return left;
        }
        root->left=trimBST(root->left,low,high);
        root->right = trimBST(root->right,low,high);
        return root;
    }
};
```

### 27.将有序数组转换为二叉搜索树

[108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

这里转换为一棵平衡的二叉搜索树，由前面的构造二叉树可知，中间位置为root，直接构造即可

```c++
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return traverse(nums,0, nums.size()-1);
    }
    TreeNode* traverse(vector<int> &nums, int l, int r) {
        if(l > r) return nullptr;
        int len = (l+r)/2;
        TreeNode *node = new TreeNode(nums[len]);
        
        node->left = traverse(nums, l ,len-1);
        node->right = traverse(nums,len + 1, r);
        return node;
    }
};
```

### 28.把二叉搜索树转换为累加树

[538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

累加树是指：

使得树中每个结点的值＝原树中大于等于原值的结点的值的和

所以即为右侧结点的和，所以我们先遍历右边节点，使用pre记录和即可。

```c++
class Solution {
public:
    TreeNode *pre = new TreeNode(0);
    TreeNode* convertBST(TreeNode* root) {
        traverse(root);
        return root;
    }
    void traverse(TreeNode *root) {
        if(root == nullptr) return;
        traverse(root->right);
        root->val += pre->val;
        pre = root;
        traverse(root->left);
    }
};
```



## 回溯算法

### 1.组合

[77. 组合](https://leetcode-cn.com/problems/combinations/)

标准回溯问题

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    vector<vector<int>> combine(int n, int k) {
        traverse(n,k,1);
        return res;
    }
    void traverse(int n,int k,int index){
        if(temp.size () == k) {
            res.push_back(temp);
            return;
        }
        for(int i = index;i <= n;i++){
            temp.push_back(i);
            traverse(n, k ,i+1);
            temp.pop_back();
        }
    }
    
    //优化
        void traverse(int n,int k,int index){
        if(temp.size () == k) {
            res.push_back(temp);
            return;
        }
        for(int i = index;i <= n-(k-temp.size())+1;i++){
            temp.push_back(i);
            traverse(n, k ,i+1);
            temp.pop_back();
        }
    }
};
```

### 2.组合总和III

[216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    vector<vector<int>> combinationSum3(int k, int n) {
        traverse(9, k, 1, n);
        return res;
    }
    void traverse(int n, int k, int index, int remain){
        if(temp.size() == k && remain == 0) {
            res.push_back(temp);
            return;
        }
        if(remain < 0 ) return;
        for(int i = index; i<=n;i++){
            temp.push_back(i);
            traverse(n,k,i+1,remain - i);
            temp.pop_back();
        }
    }
};
```

### 3.电话号码的字母组合

[17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

多个for的回溯问题

```c++
class Solution {
public:
    string str = "";
    vector<string> vecmap = {"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    vector<string> res;
    string temp = "";
    int len = 0;
    vector<string> letterCombinations(string digits) {
        len = digits.size();
        if(len == 0) return res;
        traverse(digits, 0);
        return res;
    }
    void traverse(string &digits, int index) {
        if (index == len) {
            res.push_back(temp);
            return;
        }
        int d = digits[index] - '0';
        string letter = vecmap[d - 2];
        for(int i = 0;i < letter.size();i++){
            temp.push_back(letter[i]);
            traverse(digits,index+1);
            temp.pop_back();
        }
    }
};
```

### 4.组合总和

[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

元素可以重复出现

```c++
class Solution {
public:
    vector<int> temp;
    vector<vector<int>> res;
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        traverse(candidates,0 ,target);
        return res;
    }
    void traverse(vector<int> &candidates, int index, int remain) {
        if(remain == 0) {
            res.push_back(temp);
            return;
        }
        if(remain < 0) return;
        for(int i = index; i < candidates.size(); i++){
            temp.push_back(candidates[i]);
            traverse(candidates, i, remain - candidates[i]);  // i 不是index, 不是i+1
            temp.pop_back();
        }
    }
};
```



### 5.组合总和II

[40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

这里的结果中元素不能重复出现，但candidates中有重复的元素，并且最终结果中不能有重复的集合

对于不重复的集合，如果candidates中元素不重复，则直接使用i+1进行遍历即可。

如果有重复的元素，则对于相同的元素，在当前区间中我们只作为【初始候选】一次，得到的结果就是不重复的。

排除相同的元素，我们只需要排序，然后判断即可。

```c++
class Solution {
public:
    vector<int> temp;
    vector<vector<int>> res;
    // map<int,int> cnt;
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        traverse(candidates, 0, target);
        return res;
    }

    void traverse(vector<int> &candidates, int index, int remain) {
        if(remain == 0) {
            res.push_back(temp);
            return;
        }
        if(remain < 0) return;
        for(int i = index; i < candidates.size(); i++){
            if(i!= index && candidates[i] == candidates[i-1]) continue;
                temp.push_back(candidates[i]);
                traverse(candidates, i+1, remain - candidates[i]);  // i 不是index, 不是i+1
                temp.pop_back();
        }
    }
}
```

### 6.分割回文串

[131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

首先我们需要一个judge函数实现回文子串的判断（双指针）

当当前分割是回文子串时，才进行下一个阶段的分割；

当分割点已经超过字符串的长度，表明这次分割完成，把分割的结果放入res中

```c
class Solution {
public:
    vector<string> temp;
    vector<vector<string>> res;
    vector<vector<string>> partition(string s) {
        traverse(s, 0);
        return res;
    }
    void traverse(string &s,int index){
        if(index >= s.size()) {
            res.push_back(temp);
            return;
        }
        for(int i = index; i < s.size(); i++){
            if(judge(s,index,i)){
                string str = s.substr(index, i - index + 1);
                temp.push_back(str);
                traverse(s, i+1);
            	temp.pop_back();
            }else {
                continue;
            }
        }
    }
    bool judge(string &s, int start, int end){
        for(int i = start, j = end; i < j; i++, j--){
            if(s[i] != s[j]) return false;
        }
        return true;
    }

};
```

### 7.复原IP地址

[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

有点类似分割回文子串问题，这里要在原字符串中插入"."，使得成为ip地址的形式

也就是说要对原字符串进行改变（insert），所以最后还要erase

同样的，我们需要一个judge函数，对分割的某一段是否合法进行判断

只有当前分割合法的情况下，我们才进行下一个分割

当分割点达到3个时，进行最后一部分是否合法的判断；如果合法就存进res

```c++
class Solution {
public:
    vector<string> res;
    vector<string> restoreIpAddresses(string s) {
        traverse(s, 0, 0);
        return res;
    }
    void traverse(string &s,int index, int cntPoint){
        if(cntPoint == 3) {
            if(judge(s, index, s.size()-1)){
                res.push_back(s);
            }
            return;
        }
        for(int i = index;i < s.size(); i++){
            if(judge(s, index, i)) {
                s.insert(s.begin()+i+1,'.');
                traverse(s, i+2, cntPoint + 1);
                s.erase(s.begin() + i + 1);
            }else continue;
        }
    }
    bool judge(string &s, int start, int end){
        if(start > end) return false;
        if (s[start] == '0' && start != end) { 
                return false;
        }
        int num = 0;
        for(int i = start; i <= end;i ++){
            if(s[i] <'0' || s[i] > '9') return false;
            num = num * 10 + (s[i] - '0');
            if(num > 255) return false;
        }
        if(num < 0 || num > 255) return false;
        return true;
    }  
};
```

### 8.子集

[78. 子集](https://leetcode-cn.com/problems/subsets/)

由于是返回该集合的所有子集，所以对于每一次迭代的结果都直接存入res中即可

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    vector<vector<int>> subsets(vector<int>& nums) {
        traverse(nums,0);
        return res;
    }
    void traverse(vector<int> &nums, int index) {
        // if(index > nums.size()){
        //     return;
        // }
        res.push_back(temp);
        for(int i = index;i < nums.size(); i++){ 
            temp.push_back(nums[i]);
            traverse(nums, i+1);
            temp.pop_back();
        }
    }
};
```

### 9.子集II

[90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

集合中有重复的元素，但是结果中不能有重复的集合

方法同组合总和III，只要排序过后，对重复的元素只作为初始候选一次即可。

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        traverse(nums,0);
        return res;
    }
    void traverse(vector<int> &nums, int index){
        res.push_back(temp);
        for(int i = index; i < nums.size(); i++){
            if(i != index && nums[i] == nums[i-1]) continue;
            temp.push_back(nums[i]);
            traverse(nums, i+1);
            temp.pop_back();
        }
    }
};
```

### 10.递增子序列

[491. 递增子序列](https://leetcode-cn.com/problems/increasing-subsequences/)

这里也是序列中存在重复的元素，但是结果中不能存在重复的集合

这里去重不能使用上面的方法，因为原序列不能进行排序

所以我们要考虑重复的元素在什么情况下不能再被使用？

【重复的元素在某一层的选择中，不能重复被使用】

这样我们就要在每一层的遍历中进行去重，去重的方法很多，比如使用数组

问题是这个数组应该在哪里被声明？在哪里被重置？

在遍历的过程中，一次循环其实就是一层（都是一个元素或则第n个元素），所以我们只要在循坏之前声明数组，数组在下一层中就会自动被重置

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        traverse(nums,0);
        return res;
    }
    // 去重
    void traverse(vector<int> &nums, int index){
        if(temp.size() >= 2)
            res.push_back(temp);
        int used[201]={0};  //数组的声明！！！！
        for(int i = index;i < nums.size();i ++) {
            if((!temp.empty()&&nums[i] < temp.back())||used[nums[i] + 100] == 1 ){
                continue;
            }
            used[nums[i] + 100] = 1;
            temp.push_back(nums[i]);
            traverse(nums, i + 1);
            temp.pop_back();
        }
    }
};
```

### 11.全排列

[46. 全排列](https://leetcode-cn.com/problems/permutations/)

这里由于是全排列，所以每次遍历都需要从数组nums的起点开始，这样就要考虑去重

去重就是要排除自己，我们可以使用一个数组进行去重

问题还是数组应该在哪里被声明，在哪里被重置

这里去重应该发生在树的纵向上，也就是整个排列的选择过程中。

所以如果声明在递归函数中，在选择下一层的数据时，标记数组就会被重置

所以标记数组最终声明在递归函数外，由于回溯返回时会对数据进行恢复，就不需要再进行重置。

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    int used[21]= {0};
    
    vector<vector<int>> permute(vector<int>& nums) {
        // temp.push_back(nums[0]);
        traverse(nums);
        return res;
    }
    void traverse(vector<int> &nums){
        if(temp.size() == nums.size()) {
            res.push_back(temp);
            return;
        }
        for(int i = 0; i < nums.size(); i++) {
            if(used[nums[i] + 10] == 1) continue;
            temp.push_back(nums[i]);
            used[nums[i] + 10] = 1;
            traverse(nums);
            temp.pop_back();
            used[nums[i] + 10] = 0;
        }
    }
};
```



### 12.全排列II

[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

这题的原集合中有重复的元素，结果中不允许有重复的集合

对于选择过程中排除自己，还是使用used数组，但是used数组中就不能使用used[nums[i]]了，因为有重复的元素，会把重复的元素都当作自己排除掉。这里使用used[i]即可

然后对于重复集合的去重，排序之后，对重复的元素进行过滤

但是这里的问题是，由于不使用index，而是都是使用从0开始的下标，所以会把重复的元素直接过滤掉，而不是作为【初始候选】过滤。

所以这里还要加上判断，used[i-1]==0，即前一个元素没有被选中的情况下，该元素才被过滤。

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    int used[21] = {0};
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end()); 
        traverse(nums);
        return res;
    }
    void traverse(vector<int> &nums){
        if(temp.size() == nums.size()) {
            res.push_back(temp);
            return;
        }
        for(int i = 0; i < nums.size(); i++) {
            if(i != 0 && nums[i] == nums[i-1] && used[i-1] == 0) continue;
            if(used[i] == 1) continue;
            temp.push_back(nums[i]);
            used[i] = 1;
            traverse(nums);
            temp.pop_back();
            used[i] = 0;
        }
    }
};
```





### 13.重新安排行程

[332. 重新安排行程](https://leetcode-cn.com/problems/reconstruct-itinerary/)

在回溯的过程中，由于要按照字典序进行选择，所以我们先进行排序，这里要使用自定义排序。

自定义排序的写法：

```c++
strcut mycomp{
    bool operator() (vector<string> i, vector<string> j){
        return i[1]<j[1];
    }
}
sort(a.begin(), a.end(), mycomp());
```

然后就是在回溯时，找到一个就可以返回（使用bool返回值）

```c++
class Solution {
public:
    vector<string> res;
    vector<string> finalres;
    int used[301];

    struct mycomp2 {
        bool operator() (vector<string> i, vector<string> j) {
            return (i[1] < j[1]);
        }
    };

    vector<string> findItinerary(vector<vector<string>>& tickets) {
        int len = tickets.size();
        sort(tickets.begin(), tickets.end(), mycomp2()); //自定义排序
        traverse(tickets, "JFK");
        return res;
    }
    bool traverse(vector<vector<string>> &tickets, string start) {
        if(res.size() == tickets.size()) {
            res.push_back(start);
            return true;
        }
        
        for(int i = 0;i < tickets.size(); i++){
            if(used[i] == 1) continue;
            vector<string> t = tickets[i];
            if(t[0] == start) {
                res.push_back(t[0]);
                used[i] = 1;
                if (traverse(tickets, t[1])) return true;  // 找到即返回
                res.pop_back();
                used[i] = 0;
            }
        }
        return false;
    }
};
```

### 14.N皇后

[51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

主要在遍历的过程中，行/列/对角线应该如何表示？

这里是递归每一层，每一层中for循环遍历每一列，然后对行列对角线进行判断

```c++
class Solution {
public:
    int col[20];
    int dig[20];
    int udig[20];
    vector<vector<string>> res;
    vector<string> temp;

    vector<vector<string>> solveNQueens(int n) {
        vector<string> gd(n,string(n,'.'));
        traverse(0,n,gd);
        return res;
    }
    void traverse(int c,int n,vector<string> &gd) {  // c是行数，for遍历每列
        if (c == n) {
            res.push_back(temp);
            return;
        }
        for(int i = 0;i < n;i ++){
            if(col[i] == 1 || dig[i+c] == 1 || udig[i - c + n] == 1) continue;
            gd[c][i] = 'Q';
            col[i] = 1;
            dig[i+c] = 1;
            udig[i - c + n] = 1;
            temp.push_back(gd[c]);
            traverse(c+1, n, gd);
            temp.pop_back();
            col[i] = 0;
            dig[i+c] = 0;
            udig[i - c + n] = 0;
            gd[c][i] = '.';
        }
    }
};
```

### 15.解数独

[37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

对于条件的判断是很容易想到的，使用3个二维数据来存储当前的数据使用情况。

主要是在回溯的过程中，如何判断当前是否已经找到一个可行解？？？

【当前解法】是通过如果某一个点1-9都不能放置时，说明找不到解；如果最后全部都正确放置（continue），说明找到了解。

【其他解法】有一种传参行，列。在函数开始的时候对行，列参数进行处理，找到应该放置的点的坐标。回溯时行列参数不变。

```c++
class Solution {
public:
    int row[10][10]= {0};
    int col[10][10] = {0};
    int block[10][10] = {0};
    void solveSudoku(vector<vector<char>>& board) {
        for(int i = 0;i < board.size(); i++){
            for(int j = 0; j < board[i].size(); j++){
                if(board[i][j] != '.'){
                    row[i][board[i][j] - '0'] = 1;
                    col[j][board[i][j] - '0'] = 1;
                    block[i/3*3+j/3][board[i][j] - '0'] = 1;
                }
            }
        }
        traverse(board);        
    }
    bool traverse(vector<vector<char>>& board){ 
        for(int r =0; r < 9; r++){ //r->row
            for(int i = 0;i < 9; i++){ //i->col 
                if(board[r][i] != '.') continue; //continue就不会返回false
                for(int j = 1; j <= 9; j++){ //j->num
                    if(row[r][j] == 1 || col[i][j] == 1 || block[r/3*3+i/3][j] == 1) continue;
                    board[r][i] = (j + '0');
                    row[r][j] = 1;
                    col[i][j] = 1;
                    block[(r/3)*3+i/3][j] = 1;
                    if (traverse(board)) return true;
                    row[r][j] = 0;
                    col[i][j] = 0;
                    block[(r/3)*3+i/3][j] = 0;
                    board[r][i] = '.';
                }
                return false;
            } 
        } 
        return true;
    }
};

```

## 贪心算法

### 1.分发饼干

[455. 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

把第一个（最小的）大于需求的饼干分给孩子即可。

```C++
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int j = 0;
        int res = 0;
        for(int i = 0;i < g.size(); i++) {
            while(j < s.size() && s[j] < g[i] ) j++;
            if(j < s.size()) {
                res++;
                j++;
            }
        }
        return res;
    }
};
```



### 2.摆动序列

[376. 摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)

这里可以“删除”元素使得正负交替差出现

但是这里其实只要计算出现正负交替差的次数即可，不用考虑删除元素

由于presub初始为0，所以判断时要加上等于0

```C++
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        if(nums.size() < 2) return nums.size(); //2个元素得是不等的元素的时候才视为摆动序列
        int res = 1;
        int presub = 0;
        int cursub = 0;
        for(int i = 1; i < nums.size(); i++){
            cursub = nums[i] - nums[i-1];
            if ((cursub > 0 && presub <= 0) || (cursub < 0 && presub >= 0)) {
                res++;
                presub = cursub;
            }
        }
        return res;
    }
};
```

### 3.最大子数组和

[53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

【动态规划】

是一个很典型的动态规划问题

`dp[i]`表示终点是`i`时最大数组和

当下标为`i`时，要么就是子数组的最后一个元素为`nums[i]`，子序列和为`dp[j-1]+nums[i]`;要么就是`nums[i]` 为子序列的第一个元素，子序列的和为`nums[i]`

最后取`dp[i]`中的最大值。

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        int dp[len+1];
        dp[0] = nums[0];
        int res = dp[0];

        for(int j = 1;j < len;j++){
            dp[j] = max(nums[j], dp[j-1] + nums[j]);
            res = max(res, dp[j]);
        }
        return res;
    }
};
```

【贪心算法】

这里贪心的主要思想是：累加的结果大于0时，对于后面的值求和都是有益的值，就可以继续向后求和。小于0则重置即可。

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        int count = 0;
        int res = INT_MIN;
        for(int i = 0; i < len; i++) {
            count += nums[i];
            res = max(res, count);
            if(count <= 0) count = 0;
        }
        return res;
    }
};
```



### 4.买卖股票的最佳时机II

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

【贪心】

这里由于可以多次进行买卖操作，所以贪心的想法就是对于每次第二天高于第一天的时候，都进行买卖，然后即可求得最大利润。

【动规】

这里每一天的最大利润由前一天的最大利润计算得来，但由于当前天的最大利润可能在前一天的两种状态下（持有/不持有）求得，所以这里要分别表示两种状态。

> 由于每一天的最大利润仅由前一天计算得来，所以数组可以优化为一维数组。

```c++
// 贪心法
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        int res = 0;
        for(int i = 1; i< len;i++){
            res += max(prices[i] - prices[i-1], 0);
        }
        return res;   
    }
};

//动规1
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        int dp[len+1][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for(int i = 1; i < len; i++){
            dp[i][0] = max(dp[i-1][1] + prices[i], dp[i-1][0]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i]);
        }
        return max(dp[len-1][0], dp[len-1][1]);
    }
};

// 动规2
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        int dp[2];
        dp[0] = 0;
        dp[1] = -prices[0];
        for(int i = 1; i < len; i++){
            dp[0] = max(dp[1] + prices[i], dp[0]);
            dp[1] = max(dp[1], dp[0] - prices[i]);
        }
        return max(dp[0], dp[1]);
    }
};
```

### 5.跳跃游戏

[55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

记录每一步可以到达的最远距离，如果其中存在一个点可以到达最终节点，那么就是成功

```c++

class Solution {
public:
    bool canJump(vector<int>& nums) {
        int len = 1;
        for(int i = 0 ; i <= len; i++){
            if(i == 0) len -= 1;
            len = max(len, i + nums[i]);
            if((nums[i] + i) >= (nums.size()-1)){
                return true;
            }
        }
        return false;
    }
};
```

### 6.跳跃游戏II

[45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

思路也比较简单，从终点向前遍历，找到最远的能到达终点的结点，并以此为终点继续向前遍历，直到找到起点。

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int len = nums.size();
        int index = len - 1;
        int cnt = 0;
        while(index != 0){
            int temp = 0;
            for(int i = index-1; i >= 0;i --){
                if(nums[i] + i >= index){
                    temp = i;
                }
            }
            index = temp;
            cnt++;
        }
        return cnt;
    }
};
```

### 7.K次取反后最大化的数组和

[1005. K 次取反后最大化的数组和](https://leetcode-cn.com/problems/maximize-sum-of-array-after-k-negations/)

```c++
class Solution {
public:
    struct mycomp {
        bool operator() (int a, int b){
            return abs(a) > abs(b); 
        } 
    };
    int largestSumAfterKNegations(vector<int>& nums, int k) {
        // 对于全部是正数的数组而言，只需要一直对最小的正数进行符号的翻转即可
        // 对于有负数的数组，要按照绝对值的大小，从大到小依次翻转
        // 当负数翻转完之后，要找到最小的正数，这时候如果遍历一边数组就变得很麻烦
        // 所以我们一开始就按照绝对值对数组进行排序，翻转玩负数后，最后一个数即为最小的正数
        sort(nums.begin(), nums.end(), mycomp());
        for(int i = 0;i < nums.size(); i++){
            if(k == 0) break;
            if(nums[i] < 0) {
                nums[i] = 0-nums[i];
                k--;
            }
        }
        if(k > 0) {
            while(k != 0){
                nums[nums.size() - 1] = 0 - nums[nums.size() - 1];
                k--;
            }
        }
        int res =0;
        for(int n : nums) res+=n;
        return res;
    }        
};
```

### 8.加油站

[134. 加油站](https://leetcode-cn.com/problems/gas-station/)

这题就是要想明白当前起点不能到达终点时，新起点怎么计算？怎么计算能否走完一圈？

```c++
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int rest = 0;
        int currest = 0;
        int index=0;
        for(int i = 0;i < gas.size(); i++){
            rest +=  (gas[i] - cost[i]); //总的差值
            currest +=  (gas[i] - cost[i]); //当前差值
            if(currest < 0) {
                index = i+1; // index到i之间的站点都不能作为起点，因为这些都是可到达的，意味着都是有富裕的油，但是还是不能到达i+1，所以用i+1做起点
                currest = 0;
            }
        }
        return rest < 0 ? -1 : index; //总差值大于0，意味着从index出发到len的多余油量 比 0到index少的油量 要多，所以能跑玩全程
    }
};
```

### 9.分发糖果

[135. 分发糖果](https://leetcode-cn.com/problems/candy/)

要通过2次反方向的遍历，分别计算每个点的取值

```c++
class Solution {
public:
    int candy(vector<int>& ratings) {
        //space O(n)
        vector<int> candy(ratings.size(), 1);
        int res = 0;
        for (int i = 1; i < ratings.size(); i++){
            if(ratings[i] > ratings[i-1]) candy[i] = candy[i-1] + 1; //用前一个值+1，而不是自己+1
        }
        for(int i = ratings.size()-2; i>=0; i--){
            if(ratings[i] > ratings[i+1]) candy[i] = max(candy[i], candy[i+1] + 1); // 要取前一次遍历和后一个值+1的最大值
        }
        for(int i = 0; i<ratings.size();i++) res += candy[i];
        return res;
    }
};
```

### 10.柠檬水找零

[860. 柠檬水找零](https://leetcode-cn.com/problems/lemonade-change/)

```c++
class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        int five = 0;
        int ten = 0;
        for(int i =0; i<bills.size(); i++) {
            if(bills[i] == 5) {
                five++;
            }
            if(bills[i] == 10) {
                ten ++;
                five--;
                if(five < 0) return false;
            }
            if(bills[i] == 20){
                if(ten > 0) {
                    ten --;
                    five--;
                    if(five < 0) return false;
                }else{
                    five = five - 3;
                    if(five < 0) return false;
                }
            }
        }
        return true;
    }
};
```

### 11.根据身高重建队列

[406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

很容易想到要先排序，问题就是怎么排序？排序完怎么处理？

这里的`people[i][1]`表示有这么多个人比i这个人身高要高（或相等）

所以当我们把排完序的值插入到结果中的时候，应该要插入的位置就是前面有多少个人比他高（或相等）

那么我就要按照身高先排序-身高大的在前面，这样插入后也不会乱序-已插入的身高还是被未插入的身高要大；

如果身高相同的情况下，那么就要把`people[i][1]`较小的值放在前面，因为这个值也满足条件 -（前面有多少个人比他高（或相等））

```c++
class Solution {
public:
    static bool mycomp(const vector<int> &a, const vector<int> &b){
        if(a[0] == b[0]) return a[1] < b[1];
        return a[0] > b[0];
    }
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        vector<vector<int>> res;
        sort(people.begin(), people.end(), mycomp);
        for(int i = 0;i <people.size(); i++) {
            res.insert(res.begin() + people[i][1], people[i]);
        }

        return res;
    }
};
```

### 12.用最少数量的箭引爆气球

[452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

思路就是合并有overlap的区间，最后不能合并的数量就是需要的箭的数量

一开始在合并的过程中，考虑了start和end两个端点，这样存储的时候就很麻烦

但是思考过后可以发现，其实第一个的左端点是没有作用的，只要右端点和第二个的左端点有重叠，就用第一个的右端点与第二个的右端点的最小值继续即可

```c++
//v1
class Solution {
public:
    static bool cmp (const vector<int> &a, const vector<int> &b) {
        return a[0] < b[0];
    }
    int findMinArrowShots(vector<vector<int>>& points) {
        sort(points.begin(), points.end(), cmp);
        vector<vector<int>> res;
        vector<int> temp(2);
        temp[0] = 0;
        temp[1] = INT_MAX;
        for(int i = 0; i < points.size(); i++) {
            if(temp[1] >= points[i][0]) {
                temp[0] = points[i][0];
                temp[1] = min(temp[1], points[i][1]);
            }else if(temp[1] < points[i][0]) {
                res.push_back(temp);
                temp[0] = points[i][0];
                temp[1] = points[i][1];
            }
            if(i == points.size() - 1){
                res.push_back(temp);
            }
        }
        return res.size();
    }
};

//v2
class Solution {
public:
    static bool cmp (const vector<int> &a, const vector<int> &b) {
        return a[0] < b[0];
    }
    int findMinArrowShots(vector<vector<int>>& points) {
        sort(points.begin(), points.end(), cmp);
        int res = 1;  //=1！！！！
        for(int i = 0; i < points.size()-1; i++) {
            if(points[i][1] < points[i+1][0]) {
                res++;
            }else {
                points[i+1][1] = min(points[i][1], points[i+1][1]);
            }
        }
        return res;
    }
};
```

### 13.无重叠区间

[435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

思路和上题【用最少数量的箭引爆气球】类似

```c++
class Solution {
public:
    static bool cmp (const vector<int> &a, const vector<int> &b) {
        if(a[0] == b[0]) return a[1] < b[1];
        return a[0] < b[0];
    }
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), cmp);
        int res = 0;
        for(int i = 0; i < intervals.size()-1;i++){
            if(intervals[i][1] > intervals[i+1][0]){
                intervals[i+1][1] = min(intervals[i][1], intervals[i+1][1]);
                res ++;
            }
        }
        return res;
    }
};
```

### 14.划分字母区间

[763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/)

思路很容易的理解的，s[i]的最少长度为s[i]最后一次出现的下标（right）减去第一出现的下标（left）；如果这段区间的中其他的字母最后一次出现的下标更大，则right相应增大，直到遍历的i等于right表示一段字串完成。

```c++
class Solution {
public:
    vector<int> partitionLabels(string s) {
        int cnt[27];
        for(int i = 0; i < s.size(); i++){
            cnt[s[i] - 'a'] = i;  // -'a' 不是-'0'
        }
        vector<int> res;
        int right = 0;
        int left = 0;
        for(int i =0;i < s.size(); i++){
            char c = s[i];
            right = max(right, cnt[c - 'a']);
            if(i == right) {
                res.push_back(right - left + 1);
                left = i + 1;
            }
        }
        return res;
    }
};
```

### 15.合并区间

[56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

这里左端点不变，并且可以用`res.back()`取出要比较的元素！

```c++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if(intervals.empty())
            return {};
        sort(intervals.begin(),intervals.end());
        vector<vector<int>> res;
        res.push_back(intervals[0]);
        for(int i = 1;i < intervals.size(); i++) {
            if(res.back()[1] >= intervals[i][0]){
                res.back()[1] = max(intervals[i][1], res.back()[1]);
            }else {
                res.push_back(intervals[i]);
            }
        }
        return res;
    }
};
```

### 16.单调递增的数字

[738. 单调递增的数字](https://leetcode-cn.com/problems/monotone-increasing-digits/)

主要使弄清楚当高位数字大于低位数字使，我们应该如何变化？

当高位数字减1后，该高位之后的所有低位数字都应该变成9，此时的数字才是满足要求的最大数字。

所以我们在遍历的过程中记录减1的最高位即可。

```c++
class Solution {
public:
    int monotoneIncreasingDigits(int n) {
        string num = to_string(n);
        int len = num.size();
        int res = 0;
        for(int i = num.size() - 1;i > 0;i--){
            if(num[i-1] > num[i]) {
                num[i-1] --;
                len = i;  //--之后的所有数字，都为9的上海才是最大的
            }
        }
        for(int i = len; i< num.size();i++){
            num[i] = '9';
        }
        return stoi(num);
    }
};
```

### 17.买卖股票的最佳时机含手续费*

[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

这里对于当日价格 大于 前一日价格就可以记录当日利润

记录买入价格=minprice

我们考虑从第二天起，当日价格与买入价格之间的关系

- 如果当日价格小于买入价格，那么我们就更新买入价格为当前价格

- 如果当日价格大于买入价格，但是差价小于fee，那就不操作（即不卖出）
- 如果当日价格大于买入价格，且差价大于fee，那说明我们就可以在当日卖出，累加利润
    - 这里的问题是，如果出现连续的价格上涨，多次交易会引发多次手续费，所以我们应该在上升过程中的最高的价格点卖出
    - 所以我们在卖出后，把minprices = prices[i] - fee，所以在下一次卖出时就会抵扣手续费
    - 那么这里比较minprice和prices[i]的时候，就是只有prices[i]要比前一日的price小，且差距大于fee，才会更新买入价格
        - 因为如果差价小于fee，我们进行买入操作的话，增加的一次买卖操作会带来负收益。

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int res = 0;
        int minprice = prices[0];
		for(int i = 1;i < prices.size(); i++) {
            if(prices[i] < minprice) minprice = prices[i];
            if(prices[i] > minprice + fee) {
                res += prices[i] - minprice - fee;
                minprice = prices[i] - fee; //重点在于这里
            }
        }
        return res;
    }
};
```

### 18.监控二叉树

[968. 监控二叉树](https://leetcode-cn.com/problems/binary-tree-cameras/)

贪心的策略是怎样的？

这里的想法时我们要从底层向上层考虑 -> 使用后序遍历

**原因是：**我们把叶子节点的上一层放置摄像头，这样就可以监控到最多的结点，所以我们就从底层向上层递推

一个节点可能有三种状态：

 0：该节点无覆盖
 1：本节点有摄像头
 2：本节点有覆盖

分类讨论当前节点：

- 如果是空结点，视为被覆盖
- 根据左右子节点的状态进行判断
    - 如果左右子节点都是2（覆盖），则当前结点应该没有被覆盖，并且当前结点不应该放置摄像头
    - 如果左右子节点中都是摄像头，或者有一个为摄像头，说明该节点被覆盖
    - 如果左右子节点中都无覆盖，或者有一个被覆盖，或者只有一个为摄像头，为了覆盖子节点，就需要添加摄像头

最后处理完成之后，root未被覆盖（返回0），res+1

```c++
class Solution {
public:
    int res = 0;
    // 所以我们要从下往上看，局部最优：让叶子节点的父节点安摄像头，所用摄像头最少
    int minCameraCover(TreeNode* root) {
        if(traverse(root) == 0) {  // root 无覆盖
            res++;
        }
        return res;
    }
    // 0：该节点无覆盖
    // 1：本节点有摄像头
    // 2：本节点有覆盖
    int traverse(TreeNode *root) {
        if(root == nullptr) return 2;
        int left = traverse(root->left);
        int right =traverse(root->right);
        if(left == 2 && right == 2) { // 左右节点都有覆盖
            return 0;
        }
        // left == 0 && right == 0 左右节点无覆盖
        // left == 1 && right == 0 左节点有摄像头，右节点无覆盖
        // left == 0 && right == 1 左节点有无覆盖，右节点摄像头
        // left == 0 && right == 2 左节点无覆盖，右节点覆盖
        // left == 2 && right == 0 左节点覆盖，右节点无覆盖
        if(left == 0 || right == 0) {
            res++;
            return 1;
        }
        // left == 1 && right == 2 左节点有摄像头，右节点有覆盖
        // left == 2 && right == 1 左节点有覆盖，右节点有摄像头
        // left == 1 && right == 1 左右节点都有摄像头
        if(left == 1 || right == 1) {
            return 2;
        }
        return -1; //不会执行
    }
};
```



## 动态规划

### 1.斐波那契数

[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

```c++
class Solution {
public:
    int fib(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        return fib(n-1)+fib(n-2);
    }
};
```

### 2.爬楼梯

[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

**注意：**这里都是只用到了前两个元素，所以可以进行空间的优化

```c++
class Solution {
public:
    int climbStairs(int n) {
        int dp[n+2];
        dp[1] = 1;
        dp[2] = 2;
        for(int i = 3; i <= n;i++){
            dp[i] = dp[i-1] + dp[i-2]; 
        }
        return dp[n];
    }
};
```

### 3.使用最小花费爬楼梯

[746. 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

这里要注意的是，到达楼梯顶的时候，是由dp[n]或者dp[n-1]两种情况的最小值是最小花费

因为我们dp[i]表示从dp[i]向上需要的最小花费，初始化了dp[1] 与 dp[2]

**注意：**这里都是只用到了前两个元素，所以可以进行空间的优化

```c++
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        int dp[n+2];
        dp[1] = cost[0];
        dp[2] = cost[1];
        for(int i = 3;i <= n; i++)
            dp[i] = min(dp[i-1] , dp[i-2]) + cost[i-1];
        return min(dp[n],dp[n-1]);
    }
};
```

### 4.不同路径

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

值得注意的是【动态规划的空间降维版】

- 在二维数组的遍历过程中，我们可以发现，实际上，在每一行的遍历过程中，我们只用到了当前行的前一个数据，以及上一行的当前列数据
- 如果我们只用一个一维数组来表示每一行遍历的结果，也是可行的
- 因为我们在求解dp[i]时候，dp[i-1]（同一行的前一个元素）已经求解，dp[i]（上一行的同一列）已经求解，并都还保存着。
- 还有就是我们并没有优化时间复杂度，我们还是需要遍历整个二维空间，只是只需要一个一维数组来存储结果。

【数论法】

这里还有一种数论的解决方法，比较难以想到

从起点到终点，必然要走过m+n-2次；而其中也必然有向下走过m-1次，所以一共有多少种走法就是m+n-2中取m-1，变成组合数问题

时间复杂度为O(m)

在代码实现时，为了防止溢出，所以我们在计算分子的时候，先除一下分母，一直除到分母不能除了为止，再下一次计算分子。

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        int dp[m+2][n+2];
        for(int i = 1; i <= m; i++) {
            dp[i][1] = 1;
        }
        for(int i = 1; i <=n; i++) {
            dp[1][i] = 1;
        }
        for(int i = 2; i <= m;i ++) {
            for(int j = 2; j <= n;j++){
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m][n];
    }
};

//优化版
class Solution {
public:
    int uniquePaths(int m, int n) {
        int dp[m][n];        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0)
                    dp[i][j] = 1;
                else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1]; 
    }
};

//再优化-降维
class Solution {
public:
    int uniquePaths(int m, int n) {
        int dp[n+1];
        for (int i = 0; i < n; i++) dp[i] = 1;
        for(int i = 1; i < m;i++){
            for(int j = 1; j < n; j++) {
                dp[j] = dp[j-1] + dp[j];
            }
        }
        return dp[n-1];
    }
};

// 数论版
class Solution {
public:
    int uniquePaths(int m, int n) {
        long long fz = 1;
        long long fm = m-1;
        long long a = m+n-2;
        long long b = m -1;
        while(b--) {
            fz = fz * (a--);
            while(fm != 0 && fz % fm == 0) {
                fz = fz / fm;
                fm--;
            }
        }
        return fz;
    }
};
```



### 5.不同路径II

[63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

相比于不同路径，这里多了障碍物

我们在dp的过程中，遇到了障碍物就表明到达该节点的路径数量为0

同时，一定要注意dp数组的初始化，包括置为0和初始几个数据的初始化

```c++
class Solution {
public:
    int dp[102][102];
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        for(int i =0 ;i < m;i++) {
            if(obstacleGrid[i][0] != 1)
                dp[i][0] = 1;
            else break;
        }
        for(int i = 0; i<n;i++) {
            if(obstacleGrid[0][i] != 1)
                dp[0][i] = 1;
            else break;
        }
        for(int i = 1; i<m;i++){
            for(int j = 1; j < n;j++) {
                if(obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                    continue;
                }
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
```

### 6.整数拆分

[343. 整数拆分](https://leetcode-cn.com/problems/integer-break/)

这里主要是要理解dp[i]的含义，以及递推的公式

dp[i]表示i这个数字的拆分后的最大乘积

那么dp[i]怎么由前面的数据得到呢？

i可以拆分为2部分 1~k, k+1~i  （k从1到i），这两部分的最大乘积为 k*{i-k} 与 k * dp[i-k]的最大值

​	- 因为 k*{i-k} 就是拆分为2个数的情况，而 k * dp[i-k]就是k 乘上了 （i-k）所有的拆分方式里的最大值，也就是遍历了以k分割的所有情况

然后拆分1~n里的每一个数，求出最大值即可

```c++
class Solution {
public:
    int integerBreak(int n) {
        vector<int> dp(n+2);
        dp[1] = 1;
        dp[2] = 1;
        for(int i = 2; i <= n; i++) {
            for(int j = 1; j < i; j++) {
                dp[i] = max(dp[i], max(j * (i-j), dp[i - j] * j));
            }
        }
        return dp[n];

    }
};
```

### 7.不同的二叉搜索树

[96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

首先这里是二叉搜索树，所以对于[1-n]这样的数组，当取i为root时，i左边的数据就是构成左子树，右边的数据就是构成右子树

然后i为root的情况下，所有不同的二叉搜索树的数量就为 不同左子树的数量 * 不同右子树的数量

然后这里还有一点要注意的就是 1,2,3构成的不同二叉搜索树的数量 和 2,3,4构成不同二叉树的数量是相同的。

```c++
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n+1,0);
        dp[0] = 1;
        for(int i = 1; i <= n;i++) {
        	for(int j = 1; j <= i;j++) {
        		dp[i] += dp[j-1] * dp[i-j];
        	}
        }
        return dp[n];
    }
};
```

### 8.分割等和子集

[416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

主要还是`dp[i][j]`或者`dp[i]`的含义问题，到底表示什么，然后递推公式是什么？

这题是转化为类背包问题，即能否在最后一个元素找到部分前面的元素使得和为sum/2，如果能就意味着可以

这里我们可以判断`dp[i][tempsum]`是否存在，则最后返回`dp[len-1][sum/2]`

或者`dp[i][j]`表示前i个数中，最大和为j时，真正能取到的最大和是多少;那么我们最后就判断`dp[len-1][sum/2]`是否等于sum/2

```c++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for(int n: nums) sum+=n;
        if(sum%2 != 0) return false;
        vector<vector<int>> dp(nums.size()+1, vector<int>(sum/2+1, 0)); //dp[i][j] 表示0～i的数字中是否存在元素使得和为j
        if(nums[0] <= sum / 2) dp[0][nums[0]] = 1;
        for(int i = 1; i < nums.size();i ++) {
            for(int j = 0; j <= sum/2; j++) {
                dp[i][j] = dp[i-1][j] || (j-nums[i] >= 0 ? dp[i-1][j-nums[i]] : 0);
            }
        }
        return dp[nums.size()-1][sum/2];
    }
};
```

### 9.最后一块石头的重量II

[1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/)

这里也是类似的，要转化为类背包问题

即把这些石头转化为2组，求这两组的最小差值；进一步的，转化为当最大背包容量为sum/2时，能装下的最大容量（此时价值等于容量）是多少？那就是一模一样的背包问题

如果是sum/2,那就意味着最后的差值为0

如果小于sum/2,那就意味着另一部分的和大于sum/2，差值为`sum - 2 * dp[i][sum/2]`

【这种情况下，还能使用一维数组的优化】



```c++
class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        int sum = 0;
        for(int s : stones) sum += s;
        vector<vector<int>> dp(stones.size()+1, vector<int>(sum/2+1, 0));
        for(int i = 0; i < stones.size(); i++) {
            for(int j = 0; j <= sum/2; j++) {
                if(i == 0) {
                    if (j >= stones[i]) dp[i][j] = stones[i];
                }
                else if(stones[i] > j) dp[i][j] = dp[i-1][j];
                else dp[i][j] = max(dp[i-1][j], dp[i-1][j-stones[i]] + stones[i]);
            }
        }
        return sum - 2 * dp[stones.size()-1][sum/2];
    }
};


class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        int sum = 0;
        for (int s:stones) sum += s;
        int len = stones.size();
		//这里j是倒叙遍历，因为dp[j-stones]遍历的是之前的值，但是此时之前的值应该是没有更新过的（即i-1的，而不是i的）
        vector<int> dp(sum/2+1);
        for(int i =0; i < len; i++) {
            for(int j = sum/2; j >= stones[i]; j--) {
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        return sum - 2 * dp[sum/2];
    }
};
```



### 10.目标和

[494. 目标和](https://leetcode-cn.com/problems/target-sum/)

dp的思路的是正确的，但是处理的还有有点细节没处理好

`dp[i][j]`表示使用前i个数能得到和为j的等式的数量

遍历的时候，由于target可能为负，所以偏移了sum

这里要注意的是target > sum 或者 target < -sum是无解的；所以数组最大为2*sum

```c++
//dp
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = 0;
        for(int n: nums) sum += n;
        int len = nums.size();
        vector<vector<int>> dp(len+1, vector<int>(sum * 2 + 1, 0));
        dp[0][sum + nums[0]] += 1;
        dp[0][sum - nums[0]] += 1;
        for(int i = 1;i < len; i++) {
            for(int j = 0; j <= sum*2; j++) {
                if( j >= nums[i]){
                    dp[i][j] += dp[i-1][j-nums[i]];
                }
                if(j + nums[i] <= 2 * sum) {
                    dp[i][j] += dp[i-1][j+nums[i]];
                }
            }
        }
        if(sum + target > 2 * sum || sum + target < 0) return 0;
        return dp[len-1][target + sum];
    }
};

// 回溯
class Solution {
public:
    int res = 0;
    int findTargetSumWays(vector<int>& nums, int S) {
        if(nums.size() == 0) return 0;
        int len = nums.size();
        return DFS(nums,0,0,S);
        //return res;
    }
    int  DFS(vector<int>& nums,int n,int sum,int S){
        if(n == nums.size())
            return (sum == S)?1:0;
        return DFS(nums,n+1,sum + nums[n],S) + DFS(nums,n+1,sum - nums[n],S);
    }
};
```

