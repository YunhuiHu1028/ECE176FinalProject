# ECE176FinalProject

## mask_generator.py
Make a single mask for a random image, will return the original picture, picture with mask, and mask

## dataset.py
Loading data and generate random size of mask for each of them

## model.py
Three layers CNN, baseline model

## train.py
training

## model_U_net.py
Using UNet to extract features more precisely

## train_U_net.py
Using new model to learn, modified the loss function, add more weight one missing area instead of half and half

## best_preview_v2.png
Display original image, masked image, prediction image, and completed image

train_loss = 0.3666, val_loss = 0.0599
Need run more epochs e.g. 50 epochs
Not finish evaluation yet



Database: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
-> Align&Cropped Images
