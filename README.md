
# Visual-Recognition-using-Deep-Learning-2025-Spring-HW1

| StudentID |   110550067 |
| --------- | :-----|
| **Name**  |    **簡秉霖** |


## Introduction

In this assignment, we tackle image classification with 100 classes using a dataset of 20,724 training images, 300 validation images, and 2,344 test images.I use the ResNeXt architecture with pretrained weights from Torchvision. By carefully preprocessing the data and fine-tuning hyperparameters, I achieve an impressive 95% accuracy on the public set, demonstrating the strength of the ResNeXt model.

## How to install
- Python version: 3.10

- Download required packages.<br>
  `pip install -r requirements.txt`
- Check the official Pytorch website to download torch-related packages, ensuring you select the correct CUDA version (11.8 in my case). <br>
`
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
`

## How to run
- Training: <br>`python train.py`
  
- Testing: <br>`python test.py`

## Performance snapshot
![The public prediction score](snapshot/performance.png)




