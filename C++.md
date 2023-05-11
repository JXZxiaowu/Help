# C++
## 智能指针
### unqiue_ptr
```
// 手动实现：万能引用 + 完美转发
template <class T, class... Args>
unique_ptr<T> make_unique(Args&&... args){
  reutrn unique_ptr<T>(new T(std::forward(args)...));
}
```
- unique_ptr 堆内存进行唯一管理的行为，应该由程序员保证
- unique_prt 构造函数接收 T* 类型的指针，但是这样就失去了使用 unique_ptr 的作用了，如
  ```
  int *p = new int;
  unique_ptr<int> p1(p);
  // or
  unique_ptr<int> p1(new int(0));
  ```
- unique_ptr 删除了拷贝构造函数和拷贝辅助运算符，但是保留了移动拷贝赋值和移动拷贝构造函数，因此下列操作是可行的
  ```
  unique_ptr<int> p1 = make_unique<int>(0);
  unique_ptr<int> p2 = std::move(p1);
  ```
- release() 返回 T* 类型的指针，同时放弃管理权限
- reset() 回收内存，可以传入新的需要管理的内存

### shared_ptr
- make_shared<int>(0) 返回 shared_ptr 类型变量
- shared_ptr 存在拷贝构造函数和拷贝赋值运算符函数
- 没有 release 只有 reset()
### weak_ptr
- 无法操作资源，只用来查看资源是否被释放
- lock() 返回 shared_ptr 类型变量，用来进行资源的操作，如果已被释放则返回空指针
- use_count() 返回引用计数
- expired() 返回内存是否被释放

### auto_ptr C++17 delete
  
  
