# XAW-NET: A Novel Explainable Medical Image Segmentation Model
This repository contains the official codebase of XAW-Net which is a fully connected, generalised 2.5D segmentation model for various medical imaging tasks ranging from but not limited to tumor detection in breast MRI, brain MRI, and breast PET. This model can also localise fracture in cervical spine CT scans. This network is an extension of the AW-Net ( https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Pal_AW-Net_A_Novel_Fully_Connected_Attention-Based_Medical_Image_Segmentation_Model_ICCVW_2023_paper.pdf) which was accepted in ICCV workshop 2023. XAW-Net has a novel explainable algorithm, LGCAM which provides insights about the feature vectors learnt by individual blocks the model. 

## Datasets
The XAW-Net has been trained on [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/data.html), [RSNA Cervical Spine 2022](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection), [Duke Breast Cancer MRI (https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903), and [QIN Breast](https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast). 

## Model Architecture
XAW-Net has is a fully connected encoder-decoder based segmentation model. Attention gates are incorporated to improve segmentation accuracy for small lesions in multiple imaging modalities. The model provides insights on the features learnt on each block of the model with the help of LGCAM. The model architecture has been coded in python using keras and tensorflow libraries.  

## Code Implementation
