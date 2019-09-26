### Detecting Alzheimer’s Disease Using CNNs

![Illustration](https://lh4.googleusercontent.com/iBaQAtsTKfKx1YJlZHdhJDeps7e4HjEIZ8WdjHcgtmeKbqVzDlo75OwHu707RoJX7xD5g6vKLqfN244mfIT9x2eaCwB3lX7d-kY5YeDJr5tky6MXxXtsaVouQF2HirP87Fc3T84Q)

## 1.0 Introduction

Our objective it to reliably detect and diagnose Alzheimer’s Disease through MRI scans. Alzheimer’s Disease (AD) is one of the most prevalent neurodegenerative conditions with around 50 million people diagnosed worldwide [1]. The AD can be severely impact quality of life and it is an irreversible condition with no cure. The best preventative action is to delay its progression during early onset, which require readily available diagnosis from professionals. AD is diagnosed through the detection of biomarkers, quantifiable substances whose presence is indicative of a condition. One such biomarker is brain atrophy in the hippocampus region of the brain [2] which can be observed through medical imaging data, specifically MRI scans.

![healthy vs ad](https://lh6.googleusercontent.com/epBGnhx10D1TcgZDeuayp57vTo5BNE6HcUxp97lz9kVa3Bcv0oIRNVN6WnbrkbRemb3JVWKMXJ871Qhh_F-PN4nLsiKBeZyiB08HLV6m)

Magnetic Resonance Imaging (MRI) uses large magnets and radio waves to get structural information of internal organs. A T1 weighted MRI provides information about brain structure and through this modality we can observe structural atrophy and changes in the brain.

## 2.0 Instructions

Currently LoadData only puts images from same runs into their own respective folders. 
The entire dataset is around 35 GB so I will not use that one to test. I will upload an example dataset with only a handful runs.
The example csv and zip files are in our APS360 Google Drive folder.
To run, put both of the csv and zip files in the "Data" folder (you have may have to create this). 
Run LoadData from main directory.

## 3.0 Data Processing

OASIS-3 is a compilation of neuroimaging data for over 1000 patients collected across several projects. There includes 609 cognitively normal (healthy) adults, and 489 individuals with varying degrees of cognitive impairment. The neuroimaging data includes various neuroimaging modalities with longitudinal T1 MRI being one of them. Potential users will need to apply for access to the free dataset. [6]

From this image collection, we extracted longitudinal T1 MRI images and cognitive assessments of each subject. Each subject can potentially have multiple MRI sessions, organized by the date, and each session can also have more than one scans. Each of these scans/runs is considered an individual data point. To label these data points, we parsed the cognitive assessment csv and took the closest diagnosis (assessments were not necessarily done on the same date as scans) after the scan date--after, since these diagnoses are performed taking past MRI scans into consideration. We then gave each scan a 1 if they were diagnosed with AD and 0 for other diagnoses.

Each scan is a collection of 256 images of 176 x 256 size. Depending on the model used, we processed the data as a 2D 3 channel image (not RGB channels) or a 3D 1 channel image. Sizes also varied depending on the model, where in most cases, the extremities of each image were cropped in each dimension before averaging the remaining image values. We averaged to retain spatial information from each slice.

Each training data point is a tuple of the MRI scan tensor and the correct label. The entire dataset has more healthy scans (609) than AD scans (489). We augmented our test dataset to include more copies of positive data points (AD=1). We further enrich the AD data by incorporating image augmentation to the copied data. The entire data set were split into training, validation, and test data sets (~60/20/20) making sure each subject's scans are not in two different data sets.

## 4.0 Baseline Model

![baseline](https://lh4.googleusercontent.com/h1qIqPcEVc1SFZyDhuqi6vgoLqtcxaQP-2dHRaZB-Gpf18A3IeDJ00aa84bZHafsuXhskcVXBKRJFTiJc0t_IvUsDighLufs0np47DEbvzjfoog_TDTjkcqf0KbYo-EoehtK711B)

For our baseline model we trained our data on a pre-trained 2D CNN architecture, specifically VGG 11. However, we made a few adjustments to account for 3D data. First, we utilized the 3 channels to represent brain structure data at different depths. The top and bottom regions of the scans were cropped and then we divided the remaining regions into 3 chunks. We then averaged these chunks to 3 images which were used as channels to one 2D image. Because this construction of 2D images were different than the RGB 2D images used to train VGG, we did not use the pre-trained weights. Other pretrained 2D models (alexnet, resnet, etc) returned poorer results. 

As the baseline model provided decent results, we further enhanced the 2D implementation. AD points were augmented through random noise, rotation, and horizontal flips. We also utilized class-weighted loss (penalize losses for AD data points more) to account for the imbalanced data and to optimize the model to predict AD scans more correctly (increase TPR). Through these techniques we observed an increase in performance in both test accuracy (76.8%) and TPR (78%). The model starts to overfit and converges at around 11 epochs.

![train1](https://lh3.googleusercontent.com/X4fo6L007RJkkfvuQu1iktLXusepDJrAXRssaf6R2CmOvNVCZiACwIOx9zJEJzwX2fSgF_oyGCfixtMUZxvBRs2ODJNcNdxyqV3PkvZooKRqGKts3YE3exW4jKwKzV_9eF3QJ1p-)

![train2](https://lh3.googleusercontent.com/Bflpl4HC5o_IGhENCCIjZYUGupwM-1bRMUMzhSJFEpO_8-GWIVV_heZVZoq9fyTO36WvAdMsyUIag4BsI8VjPZySRuh815EhY7qKe-dPXNaw-meqEGsoy0i6vXNhvYToShczqKTm)

## 5.0 Architecture

![architecture](https://lh6.googleusercontent.com/wtLlW3B4JSgRQG9pQrOxNCQbg7R025hWkWysPqg1T8dbZydCU2lAi1rSC8LHnjKiG20hnXHGLq1iuSjLUeGR9FwTtil-XsIuYs2P2Hm83s18rRkySIx3nuYIfqLwEIv1ttxJOwEC)

Our final model is a VGG-Based 3D CNN. This neural network is similar to the one we observed in [4] and follows closely with the 2D VGG model. The model takes in 1x48x96x96 (channel, depth, height, width) images and can be broken up into 5 blocks. Each block consists of 2 convolutional layers each with a kernel size of 3x3x3 (smallest shape to retain the notion of left/right, up/down, back/forward, and center) and stride=1. Padding is set to 1 to preserve the image dimension and each convolutional layer is activated by ReLU.  Max Pooling with stride and kernel size of 2 is appended at the end of each block. The first convolutional layer in each block increases the channel sizes by two, save for the first block, while the second layer retains that channel size. Batch normalization was used with each convolutional layer. At the end of these blocks, we average pool the final feature maps to get a list of 512*1*3*3 features to send into our classifier. The classifier consists of 3 fully-connected layers, activated through ReLU and regularized by dropout, that outputs two nodes.

To train we optimized using SGD (lr=0.001) over the Cross Entropy Loss on data batched in to a size of 8. Moreover, to make sure our model could successfully diagnose AD patients, we only accepted final validation models if the difference between the true positive and true negative rate was not greater than 10%.

![Validation Accuracy](https://lh3.googleusercontent.com/MiJdyPqnoHkyUTrGJ1qQHmFHOhwAxeHbPQVAr2kHQyGDAVxGLqj9UXPrwdn7sAUStmIYXYgOLVTz3st7neT-brk4pPBWCmYfLBmifXttP5uOzW3sDTknlkAPTYRodYGFat1ULKSW)

From our validation curve, we can see that the model converges at around 30 epochs as the model starts to overfit.

The VGG-Based model resulted in the best test accuracy with the best true positive rate (TPR) among the other 2D-based 3D architectures we tried. 

| 3D Architecture | Test Accuracy | Test True Positive Rate |
| --------------- | ------------- | ----------------------- |
| VGG-Based       | 76.9%         | 75.31% |
| ResNet152-Based | 72.38%        |   60.49% |
| ResNet101-Based | 69.13%        |    64.2% |

## 6.0 Quantitative Results

| Architecture | Test Accuracy | Test True Positive Rate |
| --------------- | ------------- | ----------------------- |
| Baseline (2D-VGG)       | 71.62%         | 65.62% |
| Enhanced Baseline | 76.8%        |   78% |
| VGG-Based 3D| 76.9%        |    75.31% |

Just fine tuning the hyperparameters and data augmentation improved the 2D architecture to have similar performances as the 3D model (with the former actually diagnosing AD more accurately). We may achieve similar increases in the 3D model if we also augment 3D AD data and weigh the loss differently for each class,

If we did not prioritize high TRP, our final model produced test accuracy of **81.23%** with a decent 66.3% TPR and 84.2 true negative rate. This test accuracy is more in line with the models from the related studies. We emphasized the TPR for two reasons. TPR directly measures how well we are achieving our objective of diagnosing AD. Second, if the model always predicted healthy, the test accuracy would be upwards to 90% due to the imbalance in the test dataset.

## 7.0 Project Difficulty/Quality

The problem in itself is a difficult task as we are asking an ML model to replicate the expertise of a doctor with only MRI scans. Practitioners would use cognitive assessments, historical personal data, and biological tests in comparison.

There are many challenges in dealing with 3D data and 3D CNNs. Overfitting was a very prevalent problem in training our models. Our model consisted of a large number of parameters (>100 milion), from 3D convolutional networks, compared to the small dataset (~3000 data points). This greatly increases our generalization error as the error bound is proportional to how complex our model is (VC dimension) and inversely proportional to the number of data points we train on. Moreover, there were hardware limitations dealing with 3D data. Not only did training and image processing take longer times, but we were also forced to keep our batch sizes small (8). In addition, most of the applications of 3D CNNs involved video data rather than 3D medical scans.There were no standardized pretrained architectures we could readily use. 

There are many alternative models we could explore in the future. Support Vector Machines may be more fit for this problem where we didn’t have access to a rich dataset. Feeding the features extracted from 3D CNNs to a 2D CNN and combining information from different imaging modalities may produce more reliable models.
