# Stroke-Classification-Using-2DCNN
Stroke-Classification-Using-2-D-Convolutional-Neural-Networks

NMF Transformation for Stroke Data Classification

Overview:

This code implements a transformative approach known as Non-Negative Matrix Factorization (NMF) to convert preprocessed stroke-related tabular data into a visually interpretable 2-D feature map. NMF is chosen for its ability to decompose non-negative matrices into lower-dimensional matrices, effectively capturing latent features in the data. The non-negativity constraint aligns with the nature of the stroke-related tabular data, ensuring that the generated features and coefficients are interpretable.

NMF Transformation Architecture:

The NMF architecture is illustrated in Figure below, where the NMF formula is given by W × H = V. Here, V represents the Original or Derived Matrix, W is the Feature Matrix, and H is the Coefficient Matrix. The architecture diagram visually explains the process of transforming raw tabular data into a 2D image, leveraging the principles of NMF.

![NMF](Stroke-Classification-Using-2DCNN
/gitimage.png)



CNN Training:

Following the transformation of tabular data into 2D images, the code proceeds to train a Convolutional Neural Network (CNN) architecture. 

# Running the Code:
To run the code, simply execute either"Tabular-to-Image.py" or "Tabular-to-Image.txt" in Google Colab. Ensure that the data file is saved on your Colab drive, and update the path accordingly in the code. The transformed images will be saved in a folder named "custom_nmf_images".
