





`Kevent()`系统调用是`kqueue`的用户界面，指对`kqueue`进行添加、删除操作的用户态界面

`Kqueue`是`freebsd`内核中的一个事件队列`kernel queue`

`Kevent()`的函数签名如下，`Kevent_t`是该操作的基本事件结构，用于表明事件的类型，行为，关联描述符以及数据等

```go
// kq 标记一个内核队列（由kqueue函数返回）
// changes 可以理解为是kevent()对kqueue中事件的操作的集合
// events  可以理解为kevent会把所有事件存在这里
// timeout 超时控制
Kevent(kq int, changes, events []Kevent_t, timeout *Timespec,) (n int, err error) {
	
}

type Kevent_t struct {
  	Ident  uint64	//该事件关联的描述符，常见的有socket fd，file fd， signal fd等
  	Filter int16	//事件的类型，比如读事件EVFILT_READ，写事件EVFILT_WRITE，信号事件EVFILT_SIGNAL
  	Flags  uint16	//事件的行为，也就是对kqueue的操作，下面介绍几个常用的
                    //如EV_ADD：添加到kqueue中，EV_DELETE从kqueue中删除
                    //EV_ONESHOT：一次性或事件，kevent返回后从kqueue中删除
                    //EV_CLEAR：事件通知给用户后，事件的状态会重置，
  	Fflags uint32
  	Data   int64
  	Udata  *byte	//用户指定的数据
}
```









kqueue是UNIX常见的io多路复用技术之一，还有比如select，poll，epoll

kqueue的原理与epoll (Linux)类似

