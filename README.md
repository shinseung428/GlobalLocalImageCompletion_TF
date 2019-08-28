# Globally and Locally Consistent Image Completion

Tensorflow implementation of Globally and Locally Consistent Image Completion on [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.  
![Alt text](images/network.JPG?raw=true "network")

## What's different from the paper  
* smaller image input size (128x128)  
* smaller patch sizes  
* less number of training iteration (500,000 iterations in the paper)
* Adam optimizer used instead of Adadelta

## Requirements
* Opencv 2.4
* Tensorflow 1.4

## Folder Setting
```
-data
  -img_align_celeba
    -img1.jpg
    -img2.jpg
    -...
```


## Train
```
$ python train.py 
```

To continue training  
```
$ python train.py --continue_training=True
```

## Test  
```
$ python test.py --img_path=./data/test/test_img.jpg
```

![Alt text](images/res.gif?raw=true "result gif")  

Use your mouse to erase pixels in the image.  
When you're done, press ENTER.  
Result will be shown in few seconds.  


## Results  
![Alt text](images/res.png?raw=true "result")
