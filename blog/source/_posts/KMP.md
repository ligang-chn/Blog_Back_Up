---
title: KMP算法
copyright: true
related_posts: true
date: 2019-09-30 22:16:23
categories: 数据结构
tags: 
    - 数据结构
    - KMP
    - C++
---

@[TOC](KMP模式匹配算法)
学习到串这一章，碰到一个不太好理解的算法，记录一下。

数据结构：**串**；

字串的**定位操作**通常称为串的模式匹配，算是串中最重要的操作之一。这里主要讲一下KMP模式匹配算法（即**克努特-莫里斯-普拉特算法**）。

<!-- more -->

## 1、前缀值求解

在进行KMP算法操作之前需要求解将要匹配的字符串的**前缀值**，表现为一个前缀数组。（有些书中称为next数组）
**第一步，前缀表**
下面是即将要匹配的字符串，先写出前缀表：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190327103519635.png#pic_center =200x200)
**第二步，求出前缀值**
把每个前缀当成独立的字符串，找出最长的公共的前后缀，并且这个前后缀是比原始字符串要短；
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190327103954172.png#pic_center =500x250)
然后，将最后一行删除，因为最后的字符串就是其本身，同时在最前面添加一个-1。
这样就构成了前缀数组：-1，0，0，1，2；

**进行匹配**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190327104404752.png#pic_center =500x250)
当出现匹配失败时（如上图），查找失配位置的前缀值，比如上图匹配a和b失败，当前的前缀值是1（即图中红色的圆圈处），所以转到匹配字符串下标为1的位置（即图中绿色的圆圈处）。
此时，将P串后移，使得红叉和绿圈对齐，从这里继续开始匹配。
绿圈之前的字符不再需要匹配，因为前面一定是匹配的，不需要验证了。（这就是KMP算法相对于朴素匹配算法的优势）
**第三步，前缀实现**
下图是前缀值求解的算法图解：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190327105635605.png#pic_center =600x300)
在上图中，下标为6的位置的前缀值怎么求解？

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190327105221318.png#pic_center =600x300)
通过观察6位置之前的字符串，发现5处的前缀值为1，要使6处的前缀值为2，只有其位置的字符为B。所以，需要做的就是检查一下，6处的字符是不是B。
len——字符串达到的最大的长度；
其实现为：

```javascript
// 前缀实现
void prefix_table(char pattern[], int prefix[], int n) {
	prefix[0] = 0;
	int len = 0;
	int i = 1;
	while (i < n) {
		if (pattern[i] == pattern[len]) {
			len++;
			prefix[i] = len;
			i++;
		}
		else {
			if (len > 0)
			{
				len = prefix[len - 1];
			}
			else {
				prefix[i] = len;
				i++;
			}
		}
	}
}
```
```javascript
// 将前缀数组后后移一位，方便后面KMP算法计算
void move_prefix_table(int prefix[], int n)
{
	int i;
	for (i = n - 1; i > 0; i--)
	{
		prefix[i] = prefix[i - 1];
	}
	prefix[0] = -1;
}
```
## 2、KMP实现
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190327112544238.png#pic_center =600x300)
```javascript
void kmp_search(char text[], char pattern[])
{
	int n = strlen(pattern);  //计算字符串长度
	int * prefix = malloc(sizeof(int) * n);//内存分配
	prefix_table(pattern, prefix, n);//前缀表求解
	move_prefix_table(prefix, n);//前缀表移位
	//text[i]   ,len(text)   =m;
	//pattern[j],len(pattrn) =n;

	while (i < m) {
		if (j == n - 1 && text[i] == pattern[j])
		{
			printf("Found pattern at %d\n", i - j);
			j = prefix[j];
		}
		if (text[i] == pattern[j]) {
			i++;
			j++;
		}
		else {
			j = prefix[j];
			if (j == -1) {
				i++;
				j++;
			}

		}
	}
}
```
## 3、参考
视频：https://www.bilibili.com/video/av11866460/?spm_id_from=333.788.videocard.0
书籍：大话数据结构
