---
layout: post
title: How can we use Tiny ImageNet dataset in Pytorch?
description: This is my notes for recording how to use Tiny ImageNet dataset in Pytorch.
date: 2022-05-19
---

### Imagenet and Tiny Imagenet

Imagenet is a famous large-scale dataset, but it has been not publicly available at least two years ago. Fortunately, a subset of Imagenet, i.e., Tiny Imagenet, is available at [http://cs231n.stanford.edu/tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip). However, Tiny Imagenet dataset is not provided by Pytorch. In this blog, we will demonstrate how to use Tiny Imagenet dataset in Pytorch step by step.  


### Download Tiny Imagenet dataset 

First you can download Tiny Imagenet dataset by right clicking http://cs231n.stanford.edu/tiny-imagenet-200.zip and chosing "save as". Its size is about 236MB. Then you can save this data file on your computer. After that, you can unzip it using a software or the command

`unzip tiny-imagenet-200.zip`

Although I only ran the above on Windows 10, I believe that this command works on both Windows 10 and Linux. Based on my own experience, using a software to unzip the data file is much faster than running the command with cmd on Windows 10. By the way, normally when you open the Command Prompt on Windows, the default path is C drive. But we usually do not work on C drive, we can go to D drive by entering `D:` followed by pressing Enter key, see [this](https://www.minitool.com/news/how-to-open-drive-in-cmd.html) for more info. 


### File descriptions
- **test**: it contains a folder called *images* consisting of 10,000 .jpeg images without labels. 
- **train**: it contains 200 folders each of which is comprised of a folder called *images* and a _boxes.txt file. The names of 200 folders represent 200 classes.  The folder *images* consists of 100 .jpeg images. 
- **val**: 
    - images: 10,000 images whose labels are given in val_annotations.txt
    - val_annotations.txt: it gives the labels for those validation images in the first two columns, e.g., val_0.JPEG	n03444034
- **wnids.txt**: list of the ids used in Tiny ImageNet from the original full set of ImageNet[^1]
- **words.txt**: description of all ids of ImageNet

[^1]: [https://www.kaggle.com/competitions/tiny-imagenet/data](https://www.kaggle.com/competitions/tiny-imagenet/data)

### Pytorch dataset format
We will use [torchvision.datasets.ImageFolder](http://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) whose format is
> root/dog/xxx.png
>
> root/dog/xxy.png
>
> root/dog/[...]/xxz.png

> root/cat/123.png
>
> root/cat/nsdf3.png
>
> root/cat/[...]/asd932_.png

Each folder contains the images with the same label. It is clear that training data is already well organized.

### Formatting validation data

The method is to create 200 folders for the 200 classes of images in Tiny ImageNet and put every image into their corresponding folder. I wrote Python code for this in the following. The code is also available at [here](https://github.com/mathfirst/Tiny-ImageNet-Pytorch)


```python
import os
from shutil import move

val_dict = {}
with open('./val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        print(split_line)
        val_dict[split_line[0]] = split_line[1] # store filenames as keys and labels as values
        
cnt_folder, cnt_image=0, 0  
for file,label in val_dict.items():
    if not os.path.exists('./' + str(label)):
        cnt_folder = cnt_folder + 1 # 200 folders in total
        print("%d creating a folder named %s" %(cnt_folder,label))
        os.mkdir(label)
        
    cnt_image = cnt_image + 1 # 10,000 images in total
    print(cnt_image)
    source = './images/' + str(file) 
    dest = './' + str(label)
    # move the images in the folder of images to the folder named with their corresponding labels
    move(source,dest) 
    
os.rmdir('./images') # removing this folder is necessary because of the Pytorch dataset format
print("done")
exit()
```


I referenced [this file](https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py) when I wrote the code.

