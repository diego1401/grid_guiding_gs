from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

def create_circle_array(n, m, r_ratio):
    """
    Create an nxm array with a circle of radius r marked with 1.0 inside and 0.0 outside.
    
    Parameters:
    n (int): Number of rows.
    m (int): Number of columns.
    r_ratio (float): Ratio of the radius of the circle to the smaller dimension of the array.
    
    Returns:
    numpy.ndarray: The resulting array.
    """
    # Calculate the radius
    square_side=max(n, m)
    diag = np.sqrt(2*square_side**2)
    # radius = np.sqrt(r_ratio/2 * diag)
    radius=r_ratio/2 * diag
    # Create an nxm array of zeros
    if r_ratio >= 1.0:
        array = np.ones((n, m))
        return torch.from_numpy(array).float() + 1e-6
    array = np.zeros((n, m))
    
    center_x, center_y = m// 2,n // 2
    
    # Create a coordinate grid
    y, x = np.ogrid[:n, :m]
    
    # Calculate the distance of each point from the center
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Set the values within the radius to 1.0
    array[distance_from_center <= radius] = 1.0
    
    return torch.from_numpy(array).float() + 1e-6

def get_fourier_coefficients(image):
    fourier_image = torch.fft.fftn(image) #rfft2 applies the fourier transform to the last 2 dimensions
    return fourier_image

def main():
     
    # Load an image
    image_path = "Lenna_test_image.png"
    image = Image.open(image_path).convert("RGB")  # Convert to RGB if the image is not in RGB mode

    # Define the transformation to convert it to a tensor
    transform = transforms.ToTensor()

    # Apply the transformation
    image_tensor = transform(image).permute(1,2,0)
    # Get fourier Coefficients
    coefs = get_fourier_coefficients(image_tensor)
    coefs_shifted = torch.fft.fftshift(coefs)
    H,W,_ = coefs_shifted.shape
    
    ratios_list = [0.01,0.1,0.25,0.5,0.75,1.0]
    smoothed_images = []
    fourier_coefficients = []
    kernels = []
    for ratio in ratios_list:
        # Clip them
        filter_kernel = create_circle_array(H,W,r_ratio=ratio).unsqueeze(-1).repeat(1,1,3)
        clipped_coefs = torch.fft.ifftshift(coefs_shifted * filter_kernel)
        
        # Retrieve image
        smoothed_image = torch.real(torch.fft.ifftn(clipped_coefs))
        smoothed_images.append(smoothed_image.clamp(0,1))
        kernels.append(filter_kernel.clamp(0,1))

        fourier_coef = torch.abs((coefs_shifted * filter_kernel))/100
        
        fourier_coefficients.append(fourier_coef.clamp(0,1))

    # Plotting
    fig, axes = plt.subplots(3, len(ratios_list), figsize=(15, 5))
    fig.suptitle('Smoothed Images and Kernels for Different Ratios')

    for i, (ratio, smoothed_image,kernel,coefs) in enumerate(zip(ratios_list, smoothed_images,kernels,fourier_coefficients)):

        # Display smoothed image
        axes[0, i].imshow(smoothed_image.numpy())
        axes[0, i].set_title(f'Ratio: {ratio}')
        axes[0, i].axis('off')

        # Display filter kernel
        axes[1, i].imshow(kernel.numpy())
        axes[1, i].axis('off')

        # Display filter kernel
        axes[2, i].imshow(coefs.numpy())
        axes[2, i].axis('off')

    # plt.show()
    plt.savefig("smooth_Lenna_test_image.png")



if __name__ == '__main__':
    main()