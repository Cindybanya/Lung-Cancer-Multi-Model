# Lung Cancer Prediction Using Radiomics and End-to-End Models

This repository contains scripts for lung cancer treatment outcome prediction based on radiomics features, SAM (Segment Anything Model), and mask-based feature encoding methods.

## Table of Contents
- [Overview](#overview)
- [File Descriptions](#file-descriptions)
- [Acknowledgements](#acknowledgements)

---

## Overview
This project uses clinical data, radiomics features, and deep learning models to predict treatment outcomes for lung cancer patients, focusing on tumor size reduction by 30%.

---

## File Descriptions
1. `clean_data.py`  
   - Preprocesses clinical raw data and Pyradiomics features.  
   - Outputs: cleaned and normalized datasets.

2. `Radiomics_prediction.py`  
   - Uses radiomics features and clinical data for prediction.  
   - Outputs: performance metrics and predictions.

3. `only_sam_prediction.py`  
   - Loads pretrained SAM checkpoints for end-to-end training and prediction.  
   - Outputs: trained model and predictions.

4. `mask_based_prediction.py`  
   - Implements mask-based feature encoding for end-to-end training and prediction.  
   - Outputs: trained model and predictions.

---
## Acknowledgements
Get Medical-SAM2 checkpoints from https://github.com/SuperMedIntel/Medical-SAM2/blob/main/checkpoints/download_ckpts.sh
!chmod +x /download_ckpts.sh


