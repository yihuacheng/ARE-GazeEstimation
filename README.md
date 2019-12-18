# Asymmetric Gaze Regression


## Introduction
This is the README file for the official code associated with the ECCV2018 paper, "Appearance-Based Gaze Estimation via Evaluation-Guided Asymmetric Regression". 

Our academic paper which describe ARE-Net in detail and provides full result can be found here: \[[PAPER]( http://openaccess.thecvf.com/content_ECCV_2018/papers/Yihua_Cheng_Appearance-Based_Gaze_Estimation_ECCV_2018_paper.pdf)\].


## Usage
We also ask that you cite the associated paper if you make use of this dataset; following is the BibTeX entry:
```
@inproceedings{eccv2018_are,
Author = {Yihua Cheng and Feng Lu and Xucong Zhang},
Title = {Appearance-Based Gaze Estimation via Evaluation-Guided Asymmetric Regression},
Year = {2018},
Booktitle = {European Conference on Computer Vision (ECCV)}
}
```

## Enviroment
To using this code, you should make sure following libraries are installed first.
```
Python>=3
Tensorflow-GPU>=1.10
PyYAML==5.1
numpy, os, math etc., which can be found in the head of code.
``` 

## Code
You need to modify the **config.yaml** first especially *data/label* and *data/root* params.  
*data/label* represents the path of label file.  
*data/root* represents the path of image file.  

A example of label file is **data** folder. Each line in label file is conducted as:
```
p00/left/1.bmp p00/right/1.bmp p00/day08/0069.bmp -0.244513310176,0.0520949295694,-0.968245505778 ... ...
```
Where our code reads image data form `os.path.join(data/root, "p00/left/1.bmp")` and reads gts of gaze direction from the rest in label file.

**Train**: We use mode 1 to represent train mode. You can train model with:
```
python main.py -m 1
```

**Predict**: We use mode 2 to predict result. You can predcit result with
```
python main.py -m 2 
```
Note that, you can not get accuracy in this mode.

**Evaluate**: We use mode 3 to evaluate trained model. You can use this mode to obtain accuracy of our model.
```
python main.py -m 3
```

Meanwhile, you can also use the code like:
```
python main.py -m 13
```
to train and evaluate model together.



