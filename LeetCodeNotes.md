

## 写在前面

这次刷题按照【[代码随想录](https://www.programmercarl.com/)】的章节顺序来刷的，预计会有接近200道题。

---

## 数组

### 二分查找

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



### 移除元素

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



### 有序数组的平方

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



### 长度最小的子数组

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



### 螺旋矩阵II

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

### 移除链表元素

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



### 设计链表

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



### 反转链表

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



### 两两交换链表中的节点

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



### 删除链表的倒数第N个节点

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



### 链表相交

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



### 环形链表II

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

### 有效的字母异位词

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



###  两个数组的交集

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



### 快乐数

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



### 两数相加

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



### 四数相加II

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



### 赎金信

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



### 三数之和

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



### 四数之和

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

### 反转字符串

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



### 反转字符串II

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



### 替换空格

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



### 翻转字符串里的单词

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

### 用栈实现队列

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



### 用队列实现栈

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



### 有效的括号

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



### 删除字符串中的所有相邻重复项

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



### 逆波兰表达式求值

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



###  滑动窗口最大值

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

### 前 K 个高频元素

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

### 二叉树的递归遍历

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



### 二叉树的迭代遍历*

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
    while(!st.empty() || cur != nullptr){
        while(cur != nullptr) {
            st.push(cur);
            cur = cur->left;
        }
        cur = st.top();
        if(cur -> right && cur->right != pre) cur = cur->right;
        else{
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



### 二叉树的层次遍历

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



### 翻转二叉树

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



### 对称二叉树

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

### 二叉树的最大深度

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

### 二叉树的最小深度

[111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

见【二叉树的层次遍历】----主要是递归法



### 完全二叉树的节点个数

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



### 平衡二叉树

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



### 二叉树的所有路径

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





### 相同的树

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



### 另一棵树的子树

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





### 左叶子之和

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



### 找树左下角的值

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



### 路径总和

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



### 从遍历序列构造二叉树

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

### 最大二叉树

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

### 合并二叉树

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

### 二叉搜索树中的搜索

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

### 验证二叉搜索树

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



### 二叉搜索树的最小绝对差

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

### 二叉搜索树中的众数

[501. 二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)

对于任意的二叉树而言，我们只需要统计二叉树的元素出现的次数即可

对于二叉搜索树而言，

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

