# 第二章 NLP前置技术解析
## 搭建Python的开发环境
### Python的优势
对于学习和从事自然语言处理工作来说，Python具有几大优势：  
1.提供丰富的自然语言处理库。  
2.编程语法相对简单(尤其易于理解)  
3.具有很多数据科学相关的库。  
### Python的科学计算发行版——Anaconda
Anaconda是一个用于科学计算的Python发行版，支持Linux系统、Mac、Windows版，它提供了包管理与环境管理的功能，可以很方便地解决多版本Python并存、切换以及各种第三方包安装问题。Anaconda能让你在数据科学的工作中轻松安装经常使用的程序包。你还可以使用它创建虚拟环境，以便更轻松地处理多个项目。Anaconda简化了工作流程，并且解决了多个包和Python版本之间遇到的大量问题。
## 正则表达式在NLP的基本应用
正则表达式是一种定义了搜索模式的特征序列，主要是用于字符串的模式匹配，或是字符的匹配。随着计算机的普及以及互联网的发展，大量的信息以电子文档方式呈现在人们面前。NLP通常所需要处理的语料一部分来自于web网页的信息抽取，一部分来自于文本格式的文档。Web网页具有很强的开发价值，具有时效性强，信息量大，结构稳定，价值高等特点，文本格式的文档多来源于人为编写或系统生成，其中包含了其中非结构化文本、半结构化文本以及结构化文本。正则表达式的作用之一是将这些文档内容从非结构化转为结构化以便后续的文本挖掘。正则表达式的另一个作用是去除“噪声”。
正则表达式是处理NLP的最基本手段之一，学习与掌握正则表达式在Python中的应用，可以帮助我们在各式复杂的文本中抽取所需要的文本信息。
### 匹配字符串
在Python中，我们会使用re模块来实现正则表达式。
- 例一：获取包含“爬虫”关键字的句子
```
import re
text_string = '文本最重要的来源无疑是网络。我们要把网络中的文本获取形成一个文本数据库。利用一个爬虫抓取到网络中的信息。爬取的策略略有广度爬取和深度爬取。根据用户的需求，爬虫可以有主题爬虫和通用爬虫之分。'
regex = '爬虫'
p_string = text_string.split('。')
for line in p_string:
    if re.search(regex,line) is not None:
        # search方法是用来查找匹配当前行是否匹配这个regex，返回的是一个match对象
        print(line) # 如果匹配到，打印这行信息

-- 上述代码输出结果
利用一个爬虫抓取到网络中的信息
根据用户的需求，爬虫可以有主题爬虫和通用爬虫之分
```
- 例二：匹配任意一个字符
```
import re
text_string = '文本最重要的来源无疑是网络。我们要把网络中的文本获取形成一个文本数据库。利用一个爬虫抓取到网络中的信息。爬取的策略略有广度爬取和深度爬取。根据用户的需求，爬虫可以有主题爬虫和通用爬虫之分。'
regex = '爬.'
p_string = text_string.split('。')
for line in p_string:
    if re.search(regex,line) is not None:
        # search方法是用来查找匹配当前行是否匹配这个regex，返回的是一个match对象
        print(line) # 如果匹配到，打印这行信息

-- 上述代码输出结果
利用一个爬虫抓取到网络中的信息
爬取的策略略有广度爬取和深度爬取
根据用户的需求，爬虫可以有主题爬虫和通用爬虫之分
```
- 例三：匹配起始和结尾字符串
```
import re
text_string = '文本最重要的来源无疑是网络。我们要把网络中的文本获取形成一个文本数据库。利用一个爬虫抓取到网络中的信息。爬取的策略略有广度爬取和深度爬取。根据用户的需求，爬虫可以有主题爬虫和通用爬虫之分。'
regex = '^文本'
p_string = text_string.split('。')
for line in p_string:
    if re.search(regex,line) is not None:
        # search方法是用来查找匹配当前行是否匹配这个regex，返回的是一个match对象
        print(line) # 如果匹配到，打印这行信息
        
-- 上述代码输出结果
文本最重要的来源无疑是网络
```
- 例四：使用中括号匹配多个字符
```
import re
text_string = ['[重要的]今年第七号台风23日登陆广东东部沿海地区','上海发布车库销量监管通知：违规者暂停网签资格','[紧要的]中国对印连发强硬信息，印度急切需要结束对峙']
regex = '^\[[重紧]..\]'
for line in text_string:
	if re.search(regex, line) is not None:
		print(line)
	else:
		print('not match')
-- 上述代码输出结果
[重要的]今年第七号台风23日登陆广东东部沿海地区
not match
[紧要的]中国对印连发强硬信息，印度急切需要结束对峙
```
### 使用转义符
与大多数编程语言相同，正则表达式里面使用“\”作为转义字符，这就可能造成反斜杠困扰。假如你需要匹配文本中的字符“\”，那么使用编程语言表示的正则表达式里将需要4个反斜杠“\\\\”：前两个和后两个分别用于在编程语言转义成反斜杠，转换成两个反斜杠再在正则表达式里转义成一个反斜杠。Python里的原生字符串很好地解决了这个问题，这个例子中的正则表达式可以使用r“\\”表示。
- 例一：使用转义符
```
import re
if re.search("\\\\", 'I have one nee\dle') is not None:
  print("match it")
else:
  print("not match")
-- 上述代码输出结果
match it
```
- 例二：使用原生字符
```
import re
if re.search(r"\\", 'I have one nee\dle') is not None:
  print("match it")
else:
  print("not match")
-- 上述代码输出结果
match it
```
### 抽取文本中的数字
1.通过正则表达式匹配年份
“[0-9]”代表的是从0到9的所有数字，那相对的“[a-z]”代表的是从a到z的所有小写字母。
- 例一：通过正则表达式匹配年份
```
import re
strings = ['War of 1812', 'There are 5280 feet to a mile', 'Happy New Year 2016!']
year_strings = []
for string in strings:
	if re.search('[1-2][0-9]{3}', string):
		# 字符串有英文有数字，匹配其中的数字部分，并且是在1000~2999之间，{3}代表的是重复之前的[0-9]三次，是[0-9] [0-9] [0-9]的简化写法。
		year_strings.append(string)
print(year_strings)
-- 上述代码输出结果
['War of 1812', 'Happy New Year 2016!']
```
2.抽取所有的年份
我们使用Python中的re模块的另一个方法findall()里来返回匹配带正则表达式的那部分字符串。
- 例二：抽取所有的年份
```
import re
years_string = '2016 wae a good year, but 2017 will be better!'
years = re.findall('[2][0-9]{3}', years_string)
print(years)
-- 上述代码输出结果
['2016', '2017']
```
## Numpy使用详解
Numpy(Numerical Python的简称)是高性能科学计算和数据分析的基础包，提供了矩阵运算的功能。Numpy提供了以下几个主要功能：  
1.ndarray——一个具有向量算术运算和复杂广播能力的多维数组对象。  
2.用于对数组数据进行快速运算的标准数学函数。  
3.用于读写磁盘数据的工具以及用于操作内存映射文件的工具。  
4.非常有用的线性代数，傅里叶变换和随机数操作。  
5.用于集成C/C++和Fortran代码工具。
### 创建数组
在Numpy中，最核心的数据结构是ndarray，ndarray代表的是多维数组，数组指的是数据的集合。
- 例一：创建数组
```
import numpy as np
vector = np.array([1,2,3,4]) # 通过Numpy中的array()，可以将向量直接导入
matrix = np.array([[1, 'Tim'], [2, 'Joey'], [3, 'Johnny'], [4, 'Frank']]) # 通过Numpy中的array()方法，也可以将矩阵导入
```
### 获取Numpy中数组的维度
首先我们通过Numpy中的一个方法arange(n)，生成0到n-1的数组。比如我们输入np.arange(15)，可以看到返回的结果是array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])，之后再通过Numpy中的reshape(row,column)方法，自动构架一个多行多列的array对象，通过Numpy提供的shape属性获取Numpy数组的维度。
- 例一：获取Numpy中数组的维度
```
import numpy as np
a = np.arange(15).reshape(3,5)
print(a.shape) # 获取Numpy中数组的维度
```
### 获取本地数据
我们可以通过Numpy中genfromtxt()来读取本地的数据集。
- 例一：读取本地数据
```
import numpy as np
data = np.genfromtxt("/Users/mac/Desktop/data/price.csv", delimiter = ",")
print(data)
```
### Numpy数组索引
Numpy支持list一样的定位操作。
- 例一：Numpy数组索引
```
import numpy as np
matrix = np.array([[1,2,3],[20,30,40]])
print(matrix[0,1])
-- 上述代码输出结果
2
```
### 切片
Numpy支持list一样的切片操作
- 例一：切片
```
import numpy as np
matrix = np.array([
[5,10,15],
[20,25,30],
[35,40,45]
])
print(matrix[:,1])
print(matrix[:,0:2])
print(matrix[1:3,:])
print(matrix[1:3,0:2])
-- 上述代码输出结果
[10 25 40]
[[ 5 10]
 [20 25]
 [35 40]]
[[20 25 30]
 [35 40 45]]
[[20 25]
 [35 40]]
```
### 数组比较
Numpy强大的地方是数组或矩阵的比较，数据比较之后会产生boolean值。
- 例一：数组比较
```
import numpy as np
matrix = np.array([
[5,10,15],
[20,25,30],
[35,40,45]
])
m = (matrix == 25)
print(m)
-- 上述代码输出结果
[[False False False]
 [False  True False]
 [False False False]]
```
- 例二：数组比较
```
import numpy as np
matrix = np.array([
[5,10,15],
[20,25,30],
[35,40,45]
])
second_column_25 = (matrix[:,1] == 25)
print(second_column_25)
print(matrix[second_column_25, :])
-- 上述代码输出结果
[False  True False]
[[20 25 30]]
```
### 替代值
Numpy可以运用布尔值来替换值。
- 例一：替代值
```
import numpy as np
vector = np.array([5,10,15,20])
equal_to_ten_or_five = (vector == 10) | (vector == 5)
vector[equal_to_ten_or_five] = 50
print(vector)
-- 上述代码输出结果
[50 50 15 20]
```
- 例二：替代值
```
import numpy as np
matrix = np.array([
[5,10,15],
[20,25,30],
[35,40,45]
])
second_column_25 = (matrix[:,1] == 25)
matrix[second_column_25, 1] = 10
print(matrix)
-- 上述代码输出结果
[[ 5 10 15]
 [20 10 30]
 [35 40 45]]
```
- 例三：替代值
```
import numpy as np
matrix = np.array([
['5','10','15'],
['20','25','30'],
['35','40','']
])
second_column_25 = (matrix[:,2] == '')
matrix[second_column_25, 2] = '0'
print(matrix)
-- 上述代码输出结果
[['5' '10' '15']
 ['20' '25' '30']
 ['35' '40' '0']]
```
### 数据类型转换
Numpy ndarray数据类型可以通过参数dtype设定，而且可以使用astype转换类型。
- 例一：数据类型转换
```
import numpy as np
vector = np.array(["1", "2", "3"])
vector = vector.astype(float)
print(vector.dtype)
-- 上述代码输出结果
float64
```

### Numpy的统计计算方法
Numpy内置很多计算方法。其中最重要的统计方法有：  
1.sum()：计算数组元素的和；对于矩阵计算结果为一个一维数组，需要指定行或者列。  
2.mean()：计算数组元素的平均值；对于矩阵计算结果为一个一维数组，需要指定行或者列。  
3.max()：计算数组元素的最大值；对于矩阵计算结果为一个一位数组，需要指定行或者列。
- 例一：数组计算
```
import numpy as np
vector = np.array([5,10,15,20])
print(vector.sum())
-- 上述代码输出结果
50
```
- 例二：数组计算
```
import numpy as np
matrix = np.array([
[5,10,15],
[20,25,30],
[35,40,45]
])
print(matrix.sum(axis=1)) # 行
print(matrix.sum(axis=0)) # 列
-- 上述代码输出结果
[ 30  75 120]
[60 75 90]
```
