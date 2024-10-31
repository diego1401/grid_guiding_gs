import torch
import numpy as np
import matplotlib.pyplot as plt

# Distribution library
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
DEVICE = 'cuda'
BATCH_SIZE = 1000

def map_to_minus1_1(tensor):
    min_val = tensor.min(0)[0]
    max_val = tensor.max(0)[0]
    
    # Normalize to [0, 1]
    tensor_norm = (tensor - min_val) / (max_val - min_val)
    
    # Scale to [-1, 1]
    tensor_scaled = tensor_norm * 2 - 1
    return tensor_scaled

def create_random_gmm(N=128,dimension=2,mode='random'):
    if mode == 'random':
        mu = torch.randn(N,dimension)

    elif mode == 'circle':
        # Random circle
        theta = torch.rand(N) * 2 * torch.pi
        
        # Generate random radii with uniform distribution in the area of the circle
        r = 0.5 #* torch.sqrt(torch.rand(N))
        
        # Convert polar coordinates to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        mu = torch.stack([x,y]).T
    
    # # Normalize mu's to [-1,1] range and center them around 0
    
    mu = map_to_minus1_1(mu)
    
    comp = MultivariateNormal(mu.to(DEVICE),0.001*torch.eye(dimension).to(DEVICE))
    alpha = torch.ones(N).to(DEVICE)
    mix = Categorical(alpha,validate_args = False) # We are sure opacities when normalized give valid probability distributions
    gmm = MixtureSameFamily(mix, comp)
    return mu,gmm

def create_occupancy_at_all_levels(mus,max_depth_level,min_value,max_value):
    occupancy = {}

    for depth_level in range(1,max_depth_level):
        resolution = 2**depth_level
        grid = torch.zeros(resolution,resolution)
        current_grid_side_length = (max_value - min_value)/2**depth_level
        
        indices = (mus - min_value - current_grid_side_length/2) * 2**(depth_level-1)
        indices = indices.long()
        
        grid[indices[:,0],indices[:,1]] = 1
        occupancy[depth_level] = grid
    return occupancy

def create_and_fill_grid_from_gaussians(centers,gmm,max_depth_level = 8):
    '''
    We assume that the gmm and the centers have been normalized so that all values are in the range [-1,1] 
    and centered around 0.
    '''
    min_value = -1; max_value = 1
    # Grid to fill
    resolution = 2**max_depth_level
    grid = torch.zeros(resolution,resolution)
    print(f"Creating a grid of size {resolution}x{resolution}")

    occupancy = create_occupancy_at_all_levels(centers,max_depth_level,min_value,max_value)
    
    # Queue of (coordinates, depth_level)
    # queue = [([1,1],1)]
    queue = [([i,j],1) for i in range(2) for j in range(2)]
    visited_coordinates = 0
    while queue:
        indices, depth_level = queue.pop()
        # if grid[indices[0],indices[1]] and depth_level == max_depth_level: continue

        current_grid_side_length = (max_value - min_value)/2**depth_level
        coordinates = torch.tensor([min_value + indices[0]/2**(depth_level-1) + current_grid_side_length/2,
                                    min_value + indices[1]/2**(depth_level-1) + current_grid_side_length/2]).to(DEVICE)
        probability = torch.exp(gmm.log_prob(coordinates))
        
        if depth_level >= max_depth_level:
            # We fill the grid
            grid[indices[0],indices[1]] = probability
            visited_coordinates += 1
        else:
            # Check if the center of grid has any density
            if probability > 1e-6 or occupancy[depth_level][indices[0]][indices[1]]:
                # We add the higher details to the queue
                new_depth_level = depth_level + 1
                new_coordinates = [([indices[0]*2+i,indices[1]*2+j],new_depth_level) for i in range(2) for j in range(2)]
                queue += new_coordinates
    print(f"Visited {visited_coordinates/(grid.shape[0]*grid.shape[1])}% of the grid")
    # print(f"Sanity Check {(grid>0).sum()/(grid.shape[0]*grid.shape[1])}")
    return grid

def create_and_fill_grid_from_gaussians_cuda(centers,gmm,max_depth_level = 6):
    '''
    We assume that the gmm and the centers have been normalized so that all values are in the range [-1,1] 
    and centered around 0.
    '''
    min_value = -1; max_value = 1
    # Grid to fill
    resolution = 2**max_depth_level
    grid = torch.zeros(resolution,resolution).to(DEVICE)
    print(f"Creating a grid of size {resolution}x{resolution}")

    occupancy = create_occupancy_at_all_levels(centers,max_depth_level,min_value,max_value)
    
    # Queue of (coordinates, depth_level)
    
    constant_add = [[i,j] for i in range(2) for j in range(2)]
    constant_add_tensor = torch.tensor(constant_add).unsqueeze(0).to(DEVICE) # (1,n_cst_add,dim), in 2D (1,4,2), in 3D (1,9,3)
 
    indices = constant_add_tensor
    for depth_level in range(1,max_depth_level + 1):
        
        indices = indices.reshape(-1,2) # (n_points,dim)
        print("Depth level",depth_level,'n indices',indices.shape[0])
        # Transforming indices into coordinates wrt current grid resolution
        current_grid_side_length = (max_value - min_value)/2**depth_level
        coordinates = indices/2**(depth_level-1) + min_value + current_grid_side_length/2
        
        new_indices = []
        for i in range(0, coordinates.shape[0], BATCH_SIZE):
            # Select the batch of elements from the tensor
            batch = coordinates[i:i + BATCH_SIZE].to('cuda')
            batch_indices = indices[i:i + BATCH_SIZE].to('cuda')

            probability = torch.exp(gmm.log_prob(batch))
            
            if depth_level >= max_depth_level:
                grid[batch_indices[:,0],batch_indices[:,1]] = probability
            else:
                # Check if the center of grid has any density
                probability_mask = probability > 1e-6
                query_grid = occupancy[depth_level].to(DEVICE)
                is_in_cell_mask = query_grid[batch_indices[:,0],batch_indices[:,1]]
                # print(f'Probability Mask stats | n_valid {probability_mask.sum()}| shape {probability_mask.shape}  | device {probability_mask.device} ')
                # print(f'is_in_cell Mask stats | n_valid {is_in_cell_mask.sum()} | shape {is_in_cell_mask.shape}  device {is_in_cell_mask.device} ')
                valid_mask = torch.logical_or(probability_mask,is_in_cell_mask)
                valid_indices = batch_indices[valid_mask].unsqueeze(1) # (n_valid_points, 1, dim)
                
                indices_to_add = valid_indices * 2 + constant_add_tensor
                new_indices.append(indices_to_add)
        if depth_level < max_depth_level:
            indices = torch.cat(new_indices,dim=0)
                
    percentage = (grid>0).sum()/(grid.shape[0]*grid.shape[1])
    print(f"Percentage of visited cells: {round(percentage.item(),2)}%")
    return grid.cpu().numpy()

def create_and_fill_grid_around_gaussians(centers,gmm,max_depth_level = 6):
    '''
    We assume that the gmm and the centers have been normalized so that all values are in the range [-1,1] 
    and centered around 0.
    '''
    min_value = -1; max_value = 1
    # Grid to fill
    resolution = 2**max_depth_level
    grid = torch.zeros(resolution,resolution).to(DEVICE)
    print(f"Creating a grid of size {resolution}x{resolution}")
    
    # Transform all centers to indices
    center_indices = (centers - min_value)*resolution/2
    center_indices = center_indices.int().to(DEVICE)
    
    # Expand these indices by adding neighbors
    blocks_to_advance = 20
    constant_add = [[i,j] for i in range(-blocks_to_advance,blocks_to_advance+1) for j in range(-blocks_to_advance,blocks_to_advance+1)]
    constant_add_tensor = torch.tensor(constant_add).unsqueeze(0).to(DEVICE) # (1,n_cst_add,dim), in 2D (1,4,2), in 3D (1,9,3)
    indices = center_indices.unsqueeze(1) + constant_add_tensor
    indices = indices.reshape(-1,2)
    valid_mask = torch.logical_and((0<=indices).all(1),(indices<grid.shape[0]).all(1))
    # print(f"Percentage of valid points: {valid_mask.sum()/indices.shape[0]}")
    indices = indices[valid_mask]
    # Tranform them into coordinates
    coordinates = 2*indices/resolution + min_value

    # Compute the probabilities
    probability = torch.exp(gmm.log_prob(coordinates))

    # Fill the grid
    grid[indices[:,0],indices[:,1]] = probability
    
    torch.cuda.synchronize()

    percentage = (grid>0).sum()/(grid.shape[0]*grid.shape[1])
    print(f"Percentage of visited cells: {round(percentage.item(),2)}%")
    return grid.cpu().numpy()

def visualize_grid(grid,centers,cmap='viridis', title='Probability Heatmap'):
    plt.figure(figsize=(8, 6))  # Adjust the size of the figure
    plt.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # Add a color bar on the side for reference
    cbar = plt.colorbar()
    cbar.set_label('Probability', rotation=270, labelpad=20)

    # Assuming centers is an array of shape (n, 2) where each row is [x, y]
     # Rescale the x and y coordinates to fit the grid dimensions
    # x_vals = (centers[:, 0] + 1)/2 * (grid.shape[0])
    # y_vals = (centers[:, 1] + 1)/2 * (grid.shape[0])
    # plt.scatter(x_vals, y_vals, c='red', s=10, marker='o', label='Centers')  # Plot points


    # Add labels and title
    plt.title(title, fontsize=15)
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)

    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.savefig("toy_problems/toy_example.png")

def main():
    # Get random Gaussian Mixture with centers
    centers, gmm = create_random_gmm(N=10,mode='circle')
    
    # Create and fill the grid
    grid = create_and_fill_grid_from_gaussians_cuda(centers,gmm,max_depth_level=9)
    # grid = create_and_fill_grid_around_gaussians(centers,gmm,max_depth_level=8)
    # Visualize the grid
    visualize_grid(grid,centers)


if __name__ == "__main__":
    main()