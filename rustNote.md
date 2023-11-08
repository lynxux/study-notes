---
title: Rust学习笔记（更新中）
description: 确实好难学！
tags:
  - 工作
  - Rust
image: "https://source.unsplash.com/0eqgB57xMeA/1600x900"
---

## 变量

### 声明变量
值得注意的是，声明变量默认是不可变的。
这里的赋值行为被称为绑定。
```rust
let v = 10;
```

- 想对一个变量进行修改（再次绑定），有两种方式：
```rust
// 声明为可变变量
let mut v = 10;
println!("{}", v);
v = 20;
println!("{}", v);
// 变量隐藏，每一次都会创建一个新变量并隐藏之前的绑定关系
let v2 = 10;
let v2 = v2 + 10;
```

### 变量的类型
#### 数字类型
```rust
let number: u32 = 14;

// 在 println 中将数据类型后缀添加到每个文本数字，以告知 Rust 有关数据类型的信息
println!("1 + 2 = {}",1u32 + 2);
```

#### 字符类型 char
有些语言中，char 类型是一个 8位无符号整数，但是 rust 是 Unicode编码，实际是一个21位整数，系统填充为32位
```rust
let uppercase_s = 'S'
```


#### 字符串类型 str String
> 关于 str 与 String 的具体区别，在所有权相关概念之后进行介绍
> str 可以看做字符串切片，引用字符串切片使用 &str

```rust
let char_1: char = 'S';
let string_1 = "string_1";   // 值得注意的是，这里是&str 类型，并不是String？
let String_2: &str = "ace";
```

#### 元组
这里是指集合到一个复合值中的不同类型值的分组。

```rust
let tuple_3 = ('e', 5i32, true)
// 访问元组使用索引的形式是 <tuple>.<index>
println!("Is '{}' the {}th letter of the alphabet? {}", tuple_3.0, tuple_3.1, tuple_3.2);
```

#### 结构
rust中结构分为三种：经典结构，元组结构，单元结构

```rust
// 定义
struct Student{
    name: String,
    level: u8,
    remote: bool
}
    
struct Grades(char, char, char, char, f32);

struct Unit;

// 实例化
let user_1 = Student{
    name: String::from("link"),
    remote: false,
    level: 1
};
let user_2 = Student{
    name: String::from("lun"),
    remote:true,
    level: 2
};

let mark_1 = Grades('A','A','A','A',9.9);
let mark_2 = Grades('A','C','B','A',9.1);

println!("{}, level:{}, remote:{}, Grades:{},{},{},{},avg:{}", user_1.name, user_1.level,user_1.remote, mark_1.0, mark_1.1, mark_1.2, mark_1.3, mark_1.4);
println!("{}, level:{}, remote:{}, Grades:{},{},{},{},avg:{}", user_2.name, user_2.level,user_2.remote, mark_2.0, mark_2.1, mark_2.2, mark_2.3, mark_2.4);
```



#### 枚举

注意，我们这里使用有双冒号 `::` 的语法 `<enum>::<variant>`

```RUST
// 定义
enum WebEvent {
    WELoad,
    WeKey(String, char),
    WEClick{x:i64, y:i64}
}

// 使用结构定义枚举
#[derive(Debug)]
struct KeyPress(String, char);
#[derive(Debug)]
struct MouseClick{x: i64, y:i64}
#[derive(Debug)]
enum WebEvent{
    WELoad(bool),
    WEClick(MouseClick),
    WEKeys(KeyPress)
}
// 实例化枚举
let we_load = WebEvent::WELoad(true);

let we_click = WebEvent::WEClick(
  MouseClick{ x: 100, y: 250}
);

let we_keys = WebEvent::WEKeys(
  KeyPress(
    String::from("link"),
    'c'
  )
);

//打印输出
println!("{:#?}", we_keys);
```

##### 补充：关于打印输出

我们使用`#[derive(Debug)]`来使得其在标准输出变得可以输出

debug 模式下必须使用`{:?}` 或者 `{:#?}`



#### 数组

数组的定义为`[Type; size]`

```rust
// 定义
let days = ["one", "two", "three"];
let bytes = [0; 5];

//
fn main() {
    let days = ["one", "two", "three"];
    let bytes = [0; 5];

   	println!("{}", days[0])
    println!("{:#?}", bytes)
}
```



#### 矢量

语法 `<vector><T>` 声明由泛型（未知）数据类型 `T` 组成的向量类型。 若要实际创建向量，请使用具体类型，如 `<vector>u32`（类型为 u32 的向量）或者 `<vector>String`（类型为字符串的向量）。

```rust
fn main() {
    let three_nums = vec![15,24,36];
    let zeros = vec![0;5];
    
    let mut fruit = Vec::new();
    fruit.push("apple");
    fruit.push("bunana");
    
    println!("fruit[1]: {}",fruit[1]);
    
    println!("pop: {:?}", fruit.pop());
    println!("fruits: {:#?}", fruit);
  
  // 如果访问越界编译时是不可知的，不同于数组（数组长度不可变），只会在运行访问时报错
}
```

#### 哈希

`HashMap<K,V>`

```rust
fn main() {

    use std::collections::HashMap;

    let mut reviews: HashMap<String, String> = HashMap::new();
    reviews.insert(String::from("key1"), String::from("value1"));
    reviews.insert(String::from("key2"), String::from("value2"));
    reviews.insert(String::from("key3"), String::from("value3"));
    println!("{:#?}", reviews);

    let key1 = "key1";
    let key2 = String::from("key2");
    println!("key2, {:?}", reviews.get(&key2));


    // Remove book review
    let key3: &str = "key3";
    println!("\n'{}\' removed.", key3);
    reviews.remove(key3);

    // Confirm book review removed
    println!("\nReview for \'{}\': {:?}", key3, reviews.get(key3));
}

```







## 函数

这里注意的是 rust 的函数返回值。

rust 可以隐式返回（通过不使用`;`结尾），也可以显示返回（使用 `return` 关键字，并且用`;`结尾）

```rust
fn main() {
    println!("Hello, world!");
    goodbye("123")
}
fn goodbye(message: &str) {
    println!("\n{}", message);
}

// 函数返回值
fn divide_by_5(num: u32) -> u32 {
  // 此时的返回值就是 num / 5
  num / 5
}

fn divide_by_5(num: u32) -> u32 {
  if num ==0 {
    return 0;
  }
  num / 5
}

```



## 控制结构

### if-else

不同于其他编程语言，这里的 ifelse 表达式可以用于赋值

```rust
let formal = true;
let notFormal = if formal == true {
  false
}else {
  true
};

println!("{}", notFormal)
```



### 循环

#### loop

可以看做 `for` 循环，可以使用 `break` 打断

值得注意的是，`rust` 的 `break` 语句可以同时返回一个值

```rust
loop {
    println!("We loop forever!");
    break;
}

// break 时返回一个值
let mut counter = 1;
let stop_loop = loop {
    counter  *= 2;
    if counter > 100 {
        break counter;
    }
};
println!("break loop when counter = {}", stop_loop);
```



#### while

```rust
while <表达式> {}

while counter < 5 {
  println!("{}", counter);
  counter += 1;
}
```



#### for

```rust
let big_birds = ["ostrich", "peacock", "stock"];
// bird 被称为迭代器
for bird in big_birds.iter() {
    println!("the {} is a big bird", bird);
}

// 创建 [0,5) 的迭代器
for number in 0..5 {
    println!("{}", number);
}
```



## 错误处理

### option 处理

`option<T>`枚举，用于处理可能存在或者可能为空的值。

在其他语言中通常使用 `nil` or `null` 对类型进行建模，而 rust 中使用 `Option<T>`

> 所以可以理解为，类似 vec! 这种类型，其实是 Option<vec!>

```rust
enum Option<T> {
  None,    // the value doesn't exist
  Some<T>, // the value exists
}
```

值得注意的是：对于矢量来说，如果直接用下标访问，但是当下标越界的时候就会 `panic`。如果使用 `Vec::get` 方法的话，如果下标越界就会返回`Option::None`

```rust
let big_birds = vec!["ostrich", "peacock", "stock"];

let bird = big_birds.get(4);
println!("{:?}", bird)
```



#### 模式匹配-match

利用 match 可以控制程序流

注意的是，match 里必须包含输入类型可能出现的所有值（下面的例子中，就是必须含有 Some 和 None）

```rust
let big_birds = vec!["ostrich", "peacock", "stock"];

for &index in [0,2,99].iter() {
  match big_birds.get(index) {
      Some(bird) => println!("{:?}", bird),
      None => println!("no bird"),
  }
}


let fruits = vec!["banana", "apple", "coconut", "orange", "strawberry"];
for &index in [0, 2, 99].iter() {
    match fruits.get(index) {
        Some(&"coconut") => println!("Coconuts are awesome!!!"),
        Some(fruit_name) => println!("It's a delicious {}!", fruit_name),
        None => println!("There is no fruit! :("),
    }
}
```



#### if let 表达式

`if let` 运算符可将模式与表达式进行比较。 如果表达式与模式匹配，则会执行 if 块。 

这样就可以只关注当个模式，而不关注其他的值。

```rust
let a_number:Option<u8> = Some(7);
match a_number {
    Some(7) => println!("that is my lucky number"),
    _ => {},  // 这里的使用下划线来匹配其他所有情况
}

// 以上代码可以简化为
if let Some(7) = a_number {
    println!("that is my lucky number");
}
```



#### unwrap 和 expect

使用 `unwrap` 可以访问 `Option` 类型的内部值

但是值得注意的是，如果值得类型是 `None` 时，`unwrap` 方法会 `panic`

```rust
let gift = Some("candy");
assert_eq!(gift.unwrap(), "candy");

println!("{:?}", gift.unwrap())  // candy

let empty_gift: Option<&str> = None;
assert_eq!(empty_gift.unwrap(), "candy");
```



`expect` 方法的作用和 `unwrap` 相同，但是 `expect` 提供一个参数作为 `panic` 时的提示

```rust
fn main() {
    let a = Some("value");
    assert_eq!(a.expect("fruits are healthy"), "value");

    let b:Option<&str> = None;
    b.expect("fruits are healthy"); // panic with `fruits are healthy`
}
```



所以使用这些方法是很危险的，可以考虑使用下面的方法：

- 使用 match 匹配
- 使用非 panic 方法，例如`unwrap_or` ，如果变体为 None，则会返回传入的参数

```rust
 assert_eq!(Some("dog").unwrap_or("cat"), "dog");
 assert_eq!(None.unwrap_or("cat"), "cat");  // 这里也是 true
```





### Result 类型

Result 类型用于返回和传错误的 `Result<T,E>` 枚举，定义为

```rust
enum Result<T, E> {
  Ok(T): //
  Err(E): // 
}
```

值得注意的是，`Result` 类型还具有 `unwrap` 和 `expect` 方法

Example:

```rust
fn main() {
    println!("{:?}", safe_division(9.0,3.0));
    println!("{:?}", safe_division(9.0,0.0));
}

#[derive(Debug)]
struct DivisionByZeroError;

fn safe_division(dividend: f64, divisor: f64) -> Result<f64, DivisionByZeroError> {
    if divisor == 0.0 {
        Err(DivisionByZeroError)
    }else {
        Ok(dividend / divisor)
    }
}
```



## 内存管理

### 所有权

