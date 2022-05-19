---
layout: post
title: How can we use Tiny Imagenet dataset in Pytorch?
description: This is my notes for recording how to use Tiny Imagenet dataset in Pytorch.
date: 2022-05-19
---

### Imagenet and Tiny Imagenet

<p>
Imagenet is a famous large-scale dataset, but it has been not publicly available at least two years ago. Fortunately, a subset of Imagenet, i.e., Tiny Imagenet, is available at http://cs231n.stanford.edu/tiny-imagenet-200.zip. However, Tiny Imagenet dataset is not provided by Pytorch. In this blog, we will demonstrate how to use Tiny Imagenet dataset in Pytorch step by step.  
</p>

### Download Tiny Imagenet dataset 

<p>
First you can download Tiny Imagenet dataset by right clicking http://cs231n.stanford.edu/tiny-imagenet-200.zip and chosing "save as". Its size is about 236MB. Then you can save this data file on your computer. After that, you can unzip it using a software or the following command

`unzip tiny-imagenet-200.zip`

Although I only ran it on Windows 10, I believe that this command works on both Windows 10 and Linux. Based on my own experience, using a software to unzip the data file is much faster than running the command with cmd on Windows 10. By the way, normally when you open the Command Prompt on Windows, the default path is C drive. But we usually do not work on C drive, we can go to D drive by entering `D:` followed by pressing Enter key, see [this](https://www.minitool.com/news/how-to-open-drive-in-cmd.html) for more info. 
</p>

### File descriptions
- **test**: it contains a folder called *images* consisting of 10,000 .jpeg images without labels. 
- **train**: it contains 200 folders each of which is comprised of a folder called *images* and a _boxes.txt file. The names of 200 folders represent 200 classes.  The folder *images* consists of 100 .jpeg images. 
- **val**: 
    - images: 10,000 images whose labels are given in val_annotations.txt
    - val_annotations.txt: it gives the labels for those validation images in the first two columns, e.g., val_0.JPEG	n03444034
- **wnids.txt**: list of the ids used in Tiny ImageNet from the original full set of ImageNet
- **words.txt**: description of all ids of ImageNet

