# ECE176 Final Project

This project studies the **image inpainting problem**, which tries to reconstruct the missing region of an image based on surrounding pixels.

We trained two models:

- **A simple CNN baseline**
- **A U-Net model**

Both models are trained on the **CelebA dataset**.

Images are randomly masked during training and the model learns to reconstruct the missing area.

---

## Dataset

Dataset used: **CelebA**

Images are resized to **128 × 128** before training.

Dataset split:

- **Train:** 90%
- **Validation:** 5%
- **Test:** 5%

---

## Training Result

Best training result:

- **Best epoch:** 28 / 30
- **Train loss:** 0.3236
- **Validation loss:** 0.3228

Evaluation result:

- **Test L1 Error:** 0.05376

---

## mask_generator.py

Generate a random rectangular mask for an image.

The function returns:

- original image  
- masked image  
- mask  

Mask definition:
- 1 = known area
- 0 = missing area


---

## dataset.py

Load CelebA images and generate a random mask for each image.

Each sample contains:
- masked_image
- mask
- original_image


Images are resized to **128 × 128** before training.

---

## model.py

Baseline model.

A simple **three-layer CNN** used as a basic model for image inpainting.

This model is used to test the pipeline before using a more complex architecture.

---

## train.py

Training script for the **baseline CNN model**.

Main steps:

- load dataset
- split dataset into train and validation
- train the model
- compute L1 loss on the masked region
- save the best model based on validation loss

---

## model_U_net.py

Implementation of the **U-Net architecture**.

Compared with the baseline CNN model, U-Net uses:

- encoder–decoder structure
- skip connections

This helps preserve spatial information and improves reconstruction quality.

---

## train_U_net.py

Training script for the **U-Net model**.

Changes compared with the baseline training:

- use U-Net architecture
- modify the loss function
- give more weight to the missing region

Loss function: loss = hole_loss * 6 + valid_loss


This encourages the model to focus more on reconstructing the missing pixels.

---

## evaluation.py

Load the saved best model and evaluate it on the **test dataset**.

Evaluation metric:

- **L1 reconstruction error**

Only the **missing region** is considered when computing the error.

---

## best_preview_v2.png

Visualization of model predictions.

Each row shows:

1. original image
2. masked image
3. prediction
4. completed image

This helps visually evaluate how well the model reconstructs the missing region.
