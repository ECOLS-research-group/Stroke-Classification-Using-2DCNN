import os
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

def nmf_data_to_images(file_path, output_folder_name, local_folder):
    # Load the preprocessed dataset
    df = pd.read_csv(file_path, header=None)

    # Drop NaN values
    df = df.dropna()

    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Normalize the features
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Perform NMF
    n_components = 10  # You can adjust this based on your requirements
    nmf_model = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)  # Increase max_iter for convergence
    W = nmf_model.fit_transform(X_normalized)

    # Save images
    N = len(W)

    for i in range(N):
        imgI = (W[i].reshape((2, 5, 1)) * 255).astype(np.uint8)  # Convert to 8-bit for image
        img_folder = 'stroke' if y[i] == 1 else 'no_stroke'
        
        # Create intermediate folders if they don't exist
        intermediate_folder = os.path.join(local_folder, output_folder_name, img_folder)
        os.makedirs(intermediate_folder, exist_ok=True)

        img_path = os.path.join(local_folder, output_folder_name, img_folder, f"img{i}.png")
        
        # Create a colored image with a 2x5 matrix-like structure
        img_array = np.concatenate([imgI, imgI, imgI], axis=-1)
        Image.fromarray(img_array, 'RGB').save(img_path)

if __name__ == "__main__":
    # Replace "/content/your_preprocessed_data.csv" with the actual file path of your preprocessed CSV file
    file_path = "/content/Stroke.csv"

    # Replace "/path/to/local/folder" with the actual local folder path where you want to save the images
    local_folder = "/content"

    # Specify a different folder name for NMF images
    output_folder_name = "custom_nmf_images"
    os.makedirs(os.path.join(local_folder, output_folder_name), exist_ok=True)

    # Call the function with the correct parameters
    nmf_data_to_images(file_path, output_folder_name, local_folder)
