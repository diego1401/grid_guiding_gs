import torch
import matplotlib.pyplot as plt

from .utils import check_gmm_constraints, create_sphere_array, define_gmm

'''
Utils Functions for Adapative Sampling
'''
PROBABILITY_THRESHOLD = {8: 1.0, 7:1e-1,6:1e-3,5:1e-6,4:1e-6,3:1e-6,2:1e-6}

def create_occupancy_at_all_levels(mus,max_depth_level,min_value,max_value,dim=3):
    occupancy = {}

    for depth_level in range(1,max_depth_level):
        resolution = 2**depth_level
        if dim == 2:
            grid = torch.zeros(resolution,resolution)
        elif dim == 3:
            grid = torch.zeros(resolution,resolution,resolution)
        else:
            raise ValueError(f"Dimensions dim={dim} not valid")
        current_grid_side_length = (max_value - min_value)/2**depth_level
        
        indices = (mus - min_value - current_grid_side_length/2) * 2**(depth_level-1)
        indices = indices.long()
        
        if dim == 2:
            grid[indices[:,0],indices[:,1]] = 1
        elif dim == 3:
            grid[indices[:,0],indices[:,1],indices[:,2]] = 1
        
        occupancy[depth_level] = grid
    return occupancy

def create_and_fill_grid_from_gaussians_cuda(centers,gmm,max_depth_level = 6,batch_size=1000,dim=3,compute_occupancy=False):
    '''
    We assume that the gmm and the centers have been normalized so that all values are in the range [-1,1] 
    and centered around 0.
    '''
    device = centers.device
    min_value = -1; max_value = 1
    # Grid to fill
    resolution = 2**max_depth_level
    
    if compute_occupancy:
        occupancy = create_occupancy_at_all_levels(centers,max_depth_level,min_value,max_value,dim=dim)
    
    # Queue of (coordinates, depth_level)
    
    if dim == 2:
        constant_add = [[i,j] for i in range(2) for j in range(2)]
        grid = torch.zeros(resolution,resolution).to(device)
        # print(f"Creating a grid of size {resolution}x{resolution}")
    elif dim == 3:
        constant_add = [[i,j,k] for i in range(2) for j in range(2) for k in range(2)]
        grid = torch.zeros(resolution,resolution,resolution).to(device)
        # print(f"Creating a grid of size {resolution}x{resolution}x{resolution}")
    else:
            raise ValueError(f"Dimensions dim={dim} not valid")
        
    constant_add_tensor = torch.tensor(constant_add).unsqueeze(0).to(device) # (1,n_cst_add,dim), in 2D (1,4,2), in 3D (1,9,3)
 
    indices = constant_add_tensor
    for depth_level in range(1,max_depth_level + 1):
        indices = indices.reshape(-1,dim) # (n_points,dim)
        # Transforming indices into coordinates wrt current grid resolution
        current_grid_side_length = (max_value - min_value)/2**depth_level
        coordinates = indices/2**(depth_level-1) + min_value + current_grid_side_length/2
        
        new_indices = []
        for i in range(0, coordinates.shape[0], batch_size):
            
            # Select the batch of elements from the tensor
            batch = coordinates[i:i + batch_size].to('cuda')
            batch_indices = indices[i:i + batch_size].to('cuda')

            probability = torch.exp(gmm.log_prob(batch))
            
            if depth_level >= max_depth_level:
                if dim == 2: grid[batch_indices[:,0],batch_indices[:,1]] = probability
                elif dim == 3: grid[batch_indices[:,0],batch_indices[:,1],batch_indices[:,2]] = probability
            else:
                # Check if the center of grid has any density
                probability_mask = probability > PROBABILITY_THRESHOLD[max_depth_level]
                if compute_occupancy:
                    query_grid = occupancy[depth_level].to(device)
                    if dim == 2:is_in_cell_mask = query_grid[batch_indices[:,0],batch_indices[:,1]]
                    elif dim == 3: is_in_cell_mask = query_grid[batch_indices[:,0],batch_indices[:,1],batch_indices[:,2]]
                    valid_mask = torch.logical_or(probability_mask,is_in_cell_mask)
                else:
                    valid_mask = probability_mask
                valid_indices = batch_indices[valid_mask].unsqueeze(1) # (n_valid_points, 1, dim)
                
                indices_to_add = valid_indices * 2 + constant_add_tensor
                new_indices.append(indices_to_add)
        if depth_level < max_depth_level:
            indices = torch.cat(new_indices,dim=0)
            
                
    # percentage = (grid>0).sum()/(grid.shape[0]*grid.shape[1]*grid.shape[2])
    # print(f"Percentage of visited cells: {round(percentage.item(),2)}%")
    # print(f'Number of visited cells {int((grid>0).sum())}' )
    return grid

def adaptive_fourier_loss(gaussians,clip_value=0.01,max_depth_level = 2,domain='xyz'):
        
        valid_mask = check_gmm_constraints(gaussians)

        # Get gmm
        mus = map_to_minus1_1(gaussians.get_xyz[valid_mask]) 
        grid = create_and_fill_grid_from_gaussians_cuda(mus,
                                                        define_gmm(mus,gaussians.get_covariance_matrix()[valid_mask],gaussians.get_opacity[valid_mask]),
                                                        max_depth_level=max_depth_level,
                                                        compute_occupancy=True)
        
        # Get Fourier Grid
        fourier_coefs = torch.fft.fftn(grid)
        
        # Get scaled coefficients
        fourier_grid_shifted = torch.fft.fftshift(fourier_coefs)
        
        # Clipping frequency
        filter_kernel = create_sphere_array(2**max_depth_level,r_ratio=clip_value).to("cuda")
        fourier_grid_shifted_capped = fourier_grid_shifted * filter_kernel

        # Gettting smooth Grid
        smooth_grid = torch.real(torch.fft.ifftn(torch.fft.ifftshift(fourier_grid_shifted_capped)))
        to_visualize =  torch.cat([grid.mean(0),smooth_grid.mean(0)],dim=1).clamp(0,0.3).detach().cpu().numpy()
        if domain == 'frequency':
            # Loss is applied in frequency domain
            fourier_loss = torch.abs(fourier_grid_shifted_capped - fourier_grid_shifted).mean() # Some coefs are negative. We want to make them all tend to 0.
            
            return fourier_loss, to_visualize
        elif domain == 'xyz':
            # Loss is applied in xyz domain
            fourier_loss = torch.square(smooth_grid - grid).mean()
            return fourier_loss, to_visualize
        else:
            raise ValueError(f"Domain {domain} is not implemented.")

'''
Toy functions to visualize in 2D
'''

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
    
    mu = map_to_minus1_1(mu).to('cuda')

    from torch.distributions.multivariate_normal import MultivariateNormal
    from torch.distributions.mixture_same_family import MixtureSameFamily
    from torch.distributions.categorical import Categorical
    
    comp = MultivariateNormal(mu,0.001*torch.eye(dimension).to('cuda'))
    alpha = torch.ones(N).to('cuda')
    mix = Categorical(alpha,validate_args = False) # We are sure opacities when normalized give valid probability distributions
    gmm = MixtureSameFamily(mix, comp)
    return mu,gmm

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
    grid = create_and_fill_grid_from_gaussians_cuda(centers,gmm,max_depth_level=9,dim=2,compute_occupancy=True).cpu().numpy()
    # Visualize the grid
    visualize_grid(grid,centers)


if __name__ == "__main__":
    main()