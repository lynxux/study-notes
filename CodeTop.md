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



