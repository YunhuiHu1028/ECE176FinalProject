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








#
Best epoch: Epoch=28/30 | train_loss=0.3236 | val_loss=0.3228
#
Evaluation accuracy: Test L1 Error: 0.05376023278580479



Database: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
-> Align&Cropped Images
