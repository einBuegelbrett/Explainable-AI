# Explainable-AI - Post-Hoc Analyse

## Task description
Creation of a CNN for brain tumour detection and application of Layer-wise Relevance Propagation (LRP) to analyse the relevance of each node for network decision making to find out which patterns helped in detection.

## Dataset
The dataset for training and trying can be found on [kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset?resource=download).
The dataset needs to be placed in the workng directory like this:
```
XAI-BrainTumor
├── brain-tumor-dataset
│   ├── Brain-Tumor
│   │   ├── Cancer (1).jpg
│   │   ├── Cancer (2).jpg
│   │   ...
│   └── Healthy
│       ├── Not Cancer  (1).jpg
│       ├── Not Cancer  (2).jpg
│       ...
├── main.ipynb
...
```

## Project
### Introduction
Brain tumor is an abnormal growth of cell of brain. [1]\
The brain tumor is on the right side of the brain, towards the middle and slightly back. There you can see the large, well-defined mass.\
![brain tumor](images-documentation/cancer_1.jpg "Title")

### Convolutional Neural Networks (CNN)
Brain tumor detections are using MRI images is a challenging task, because the complex structure of the brain. [1] The goal of this project is to understand, what makes it so hard to detect the tumor for an artificial intelligence, especialy a CNN.

Convolutional neural networks have been applied to a wide variety of computer vision tasks. Recent advances in semantic segmentation have enabled their application to medical image segmentation. [2]

### Layer-wise Relevance Propagation (LRP)


### Structure


### Course of the project
The next steps, i.e. building the CNN and running the LRP with analysis and conclusion, will all be done and explained in ``main.ipynb``. 

## Interesting cases to try
- Brain tumor: 101, 257, 1000, 1001, 1003, 1005, 1007, 1022, 1023, 1029, 1034, 2025, 2026, 2035, 2036
- Healthy: 4, 6

## Inspiration for the project
* [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* [PyTorchRelevancePropagation by kaifishr](https://github.com/kaifishr/PyTorchRelevancePropagation)
* [lrp_toolbox](https://github.com/sebastian-lapuschkin/lrp_toolbox)

## Literaturverzeichnis
* [1] [Image Processing Techniques for Brain Tumor Detection: A Review](https://d1wqtxts1xzle7.cloudfront.net/40014067/IJETTCS-2015-10-01-7-libre.pdf?1447569226=&response-content-disposition=inline%3B+filename%3DImage_Processing_Techniques_for_Brain_Tu.pdf&Expires=1732787103&Signature=F2~tywWaIuTf0XXNDVScYlEQgee8b1217Rm8Zhw9KqWc9CGPjEsdJloSP0STUU~0wHc6HsjsBXQbYoBZUfHDFM~YTXkZJO3-pPNGkgJQIMmlraEcINHVU0O2mMRvzkGStvPzHw5cA3QfSuYolTAxsoITc~8hGCSgYibms8EWEIBuVuU6o53qdeCkKO8hEkdJ-l7KyuyLWzd1MAWF8vDmsr7lSY9pArTw248jMknpsnblIEFWkXjYQbatFyTKPDLCaP9dbLz33qm7oDj5UQfkEVzOIRYe1Z3KO48NLRnRpB~8y7ZsVeg488171NLvyt6rAckpjkyBCSEDN8fEjfph1A__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
* [2] [CNN-based Segmentation of Medical Imaging Data](https://arxiv.org/abs/1701.03056)
* [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
* [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)

# todo: schwächen und stärken von methoden
