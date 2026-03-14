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
