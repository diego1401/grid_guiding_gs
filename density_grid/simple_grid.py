import torch 
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt

# Adaptive Sampling
from .adaptive_sampling_utils import create_and_fill_grid_from_gaussians_cuda, map_to_minus1_1
from .utils import check_gmm_constraints

class SimpleDensityGrid(nn.Module):
    def __init__(self, grid_size=0,sample_gmm:bool=False,number_of_samples:int=0):
        super().__init__()
        # assert grid_size >0
        self.grid_size = grid_size
        self.batch_size = 10
        self.coords_flat = None
        if grid_size>0:self.grid = torch.zeros(self.grid_size, self.grid_size, self.grid_size,device='cuda')
        self.coords_flat = None
        self.sample_gmm = sample_gmm
        if sample_gmm:
            assert number_of_samples>0
            self.number_of_samples = number_of_samples
    
    def define_coordinates(self,gaussians):
        x_min,x_max = gaussians.get_xyz[:,0].min(),gaussians.get_xyz[:,0].max()
        y_min,y_max = gaussians.get_xyz[:,1].min(),gaussians.get_xyz[:,1].max()
        z_min,z_max = gaussians.get_xyz[:,2].min(),gaussians.get_xyz[:,2].max()
        bounding_box_edges = (x_min,x_max,y_min,y_max,z_min,z_max)
        #  Getting coordinates
        x_min,x_max,y_min,y_max,z_min,z_max = bounding_box_edges
        x_coords = torch.linspace(x_min.item(), x_max.item(), self.grid_size)
        y_coords = torch.linspace(y_min.item(), y_max.item(), self.grid_size)
        z_coords = torch.linspace(z_min.item(), z_max.item(), self.grid_size)
        delta_x = (x_max - x_min)/self.grid_size
        delta_y = (y_max - y_min)/self.grid_size
        delta_z = (z_max - z_min)/self.grid_size
        delta = torch.tensor([delta_x,delta_y,delta_z])

        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords,z_coords, indexing='xy')  # grid_x and grid_y of shape (H,W,D)
        coords_stack = torch.stack([grid_x, grid_y,grid_z], dim=-1).reshape(-1,3)  # (H*W*D,3)
        random_displacement = torch.rand_like(coords_stack)*2 - 1.0
        coords = coords_stack + random_displacement * delta/2
        
        self.coords_flat = coords


    def apply_fourier_loss_n_iterations(self,gaussians,clip_value,n_iterations):

        self.check_gmm_constraints(gaussians)
        lr = 1e-3
        # Optimizer
        l = [
            {'params': [gaussians._xyz], 'lr': lr, "name": "xyz"},
            {'params': [gaussians._scaling], 'lr': lr, "name": "scaling"},
            {'params': [gaussians._rotation], 'lr': lr, "name": "rotation"}
        ]

        optimizer = torch.optim.Adam(l)
        
        total_log_loss = 0.0
        for _ in tqdm(range(n_iterations)):
            if not self.check_gmm_constraints(gaussians):
                l = [
                    {'params': [gaussians._xyz], 'lr': lr, "name": "xyz"},
                    {'params': [gaussians._scaling], 'lr': lr, "name": "scaling"},
                    {'params': [gaussians._rotation], 'lr': lr, "name": "rotation"}
                ]
                optimizer = torch.optim.Adam(l)

            optimizer.zero_grad()
            loss, log_loss = self.get_fourier_loss(gaussians,clip_value)
            
            total_log_loss += log_loss.item()
            loss.backward()
            # print(loss)
            # print('xyz',gaussians._xyz[0])
            optimizer.step()
        return total_log_loss / n_iterations

    def get_fourier_loss(self,gaussians,clip_value=0.1):
        if self.sample_gmm:
            return self._get_fourier_loss_sample_gmm(gaussians,clip_value)
        return self._get_fourier_loss(gaussians,clip_value)
    
    def get_fourier_loss_adaptive(self,gaussians,clip_value=0.01,max_depth_level = 2,domain='xyz'):
        
        self.check_gmm_constraints(gaussians)
        # Get gmm
        mus = map_to_minus1_1(gaussians.get_xyz) 
        grid = create_and_fill_grid_from_gaussians_cuda(mus,
                                                        define_gmm(mus,gaussians.get_covariance_matrix(),gaussians.get_opacity),
                                                        max_depth_level=max_depth_level,
                                                        compute_occupancy=True)
        
        # Get Fourier Grid
        fourier_coefs = get_fourier_coefficients(grid)
        
        # Get scaled coefficients
        fourier_grid_shifted = torch.fft.fftshift(fourier_coefs)
        
        # Clipping frequency
        filter_kernel = create_sphere_array(2**max_depth_level,r_ratio=clip_value).to("cuda")
        fourier_grid_shifted_capped = fourier_grid_shifted * filter_kernel

        if domain == 'frequency':
            # Loss is applied in frequency domain
            new_target = fourier_grid_shifted_capped
            fourier_loss = torch.abs(new_target - fourier_grid_shifted).mean() # Some coefs are negative. We want to make them all tend to 0.
            return fourier_loss, get_grid_slices(grid,fourier_grid_shifted_capped)
        elif domain == 'xyz':
            # Loss is applied in xyz domain
            smoothed_grid = torch.real(torch.fft.ifftn(torch.fft.ifftshift(fourier_grid_shifted_capped)))
            fourier_loss = torch.square(smoothed_grid - grid).mean()
            to_visualize =  torch.cat([grid.mean(0),smoothed_grid.mean(0)],dim=1).clamp(0,0.3).detach().cpu().numpy()
            return fourier_loss, to_visualize
        else:
            raise ValueError(f"Domain {domain} is not implemented.")
    
    def get_fourier_loss_adaptive_fixed_target(self,gaussians,clip_value=0.01,max_depth_level = 2,target=None):
        
        self.check_gmm_constraints(gaussians)
        # Get gmm
        mus = map_to_minus1_1(gaussians.get_xyz)
        gmm = define_gmm(mus,gaussians.get_covariance_matrix(),gaussians.get_opacity)
        
        # Normalize mus
        
        grid = create_and_fill_grid_from_gaussians_cuda(mus,gmm,max_depth_level=max_depth_level,compute_occupancy=True)
        
        # Get Fourier Grid
        fourier_coefs = get_fourier_coefficients(grid)
        average_fourier_coefficient = torch.abs(fourier_coefs).mean()
        
        # Get scaled coefficients
        fourier_grid_shifted = torch.fft.fftshift(fourier_coefs)
        # Using a clip to select the high frequencies to punish
        

        if target is None:
            # Create target
            grid_size = 2**max_depth_level
            filter_kernel = create_sphere_array(grid_size,r_ratio=clip_value).to("cuda")
            fourier_grid_shifted_capped = fourier_grid_shifted * filter_kernel
            ## Loss is applied in frequency domain
            # new_target = fourier_grid_shifted_capped
            # fourier_loss = torch.abs(new_target - fourier_grid_shifted).mean() # Some coefs are negative. We want to make them all tend to 0.
            
            # Loss is applied in xyz domain
            new_target = torch.real(torch.fft.ifftn(torch.fft.ifftshift(fourier_grid_shifted_capped)))
            fourier_loss = torch.square(new_target - grid).mean()
            
            return fourier_loss, average_fourier_coefficient, (grid>0).sum(), get_grid_slices(grid,fourier_grid_shifted_capped), new_target
        
        else:
            # Loss is applied in xyz domain
            # Target should be the smoothed grid
            fourier_loss = torch.square(target - grid).mean() # Some coefs are negative. We want to make them all tend to 0.
            
            # Loss is applied in frequency domain
            # Target should be the fourier coefficients
            # fourier_loss = torch.abs(target - fourier_grid_shifted).mean()
            
            return fourier_loss, average_fourier_coefficient, (grid>0).sum(), (grid.mean(0), None), None

    def _get_fourier_loss_sample_gmm(self,gaussians,clip_value=0.01):
        self.check_gmm_constraints(gaussians)
        # Get gmm
        gmm = define_gmm(gaussians.get_xyz,gaussians.get_covariance_matrix(),gaussians.get_opacity.detach())
        ## Sample from GMM to obtain points
        samples = gmm.sample(torch.Size([self.number_of_samples]))
        # Find the minimum and maximum values of the tensor
        min_val = samples.min()
        max_val = samples.max()

        # # Normalize the tensor to [0, 1]
        samples = (samples - min_val) / (max_val - min_val)
        sample_count = torch.zeros(self.grid_size, self.grid_size, self.grid_size,device='cuda')
        grid = torch.zeros(self.grid_size, self.grid_size, self.grid_size,device='cuda')
        
        for i in range(0, samples.shape[0], self.batch_size):
            # Select the batch of elements from the tensor
            batch = samples[i:i + self.batch_size].to('cuda')
            true_log_probs = gmm.log_prob(batch)  # (H*W)
            i,j,k = self.grid_xyz_to_index(batch)
            
            grid[i,j,k] += true_log_probs
            sample_count[i,j,k] += 1

        grid[sample_count>0] /= sample_count[sample_count>0]
        # Get Fourier Grid
        fourier_grid = torch.fft.fftn(grid) #rfft2 applies the fourier transform to the last 2 dimensions
        fourier_grid_shifted = torch.fft.fftshift(fourier_grid)
        real_fourier_loss = torch.abs(torch.real(fourier_grid_shifted)).mean()

        # Using a distance kernel to weight the frequencies
        filter_kernel = create_distance_sphere(self.grid_size).to("cuda")
        weighted_fourier_grid = fourier_grid_shifted * filter_kernel
        fourier_loss = torch.abs(torch.real(weighted_fourier_grid)).mean()
        return fourier_loss, real_fourier_loss
    
    def _get_fourier_loss(self,gaussians,clip_value=0.01):
        self.check_gmm_constraints(gaussians)
        # Get gmm
        gmm = define_gmm(gaussians.get_xyz,gaussians.get_covariance_matrix(),gaussians.get_opacity)

        ## Sample from GMM to obtain points
        # samples = gmm.sample(torch.Size([10_000]))
        # Find the minimum and maximum values of the tensor
        # min_val = samples.min()
        # max_val = samples.max()

        # # Normalize the tensor to [0, 1]
        # samples = (samples - min_val) / (max_val - min_val)
        # sample_count = torch.zeros(self.grid_size, self.grid_size, self.grid_size,device='cuda')
        # Fill the grid in a differentiable way
        
        
        self.define_coordinates(gaussians)
        samples = self.coords_flat

        grid = torch.zeros(self.grid_size, self.grid_size, self.grid_size,device='cuda')
        
        #TODO: Remove for loop
        for i in range(0, samples.shape[0], self.batch_size):
            # Select the batch of elements from the tensor
            batch = samples[i:i + self.batch_size].to('cuda')
            true_log_probs = gmm.log_prob(batch)  # (H*W)
            # i,j,k = self.grid_xyz_to_index(batch)
            i,j,k = self.grid_index_to_flat(torch.tensor([i+j for j in range(batch.shape[0])]))
            grid[i,j,k] += true_log_probs
            # sample_count[i,j,k] += 1

        # grid[sample_count>0] /= sample_count[sample_count>0]
        # Get Fourier Grid
        fourier_grid = torch.fft.fftn(grid) #rfft2 applies the fourier transform to the last 2 dimensions
        fourier_grid_shifted = torch.fft.fftshift(fourier_grid)
        real_fourier_loss = torch.abs(torch.real(fourier_grid_shifted)).mean()

        ## Using a clip to select the high frequencies to punish
        # filter_mask = ~(create_sphere_array(self.grid_size,r_ratio=clip_value)>0.1)
        # high_frequencies = fourier_grid_shifted[filter_mask]
        # fourier_loss = torch.abs(torch.real(high_frequencies)).mean()

        # Using a distance kernel to weight the frequencies
        filter_kernel = create_distance_sphere(self.grid_size).to("cuda")
        weighted_fourier_grid = fourier_grid_shifted * filter_kernel
        fourier_loss = torch.abs(torch.real(weighted_fourier_grid)).mean()
        return fourier_loss, real_fourier_loss
    
    def grid_xyz_to_index(self,xyz):
        # Normalize to [0, grid_size-1] range
        normalized_positions = (xyz + 1) * (self.grid_size - 1) / 2

        # Convert to integer indices
        grid_indices = normalized_positions.long()
        return grid_indices[:,0],grid_indices[:,1],grid_indices[:,2]

    def grid_index_to_flat(self,n):
        i = torch.div(n,(self.grid_size * self.grid_size),rounding_mode='floor')
        j = torch.div((n % (self.grid_size * self.grid_size)),self.grid_size,rounding_mode='floor')
        k = n % self.grid_size
        return i,j,k


if __name__ == '__main__':
    a = create_sphere_array(128,0.1)[64]
    print(a.shape)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(a.numpy())
    plt.savefig('testing_filter.png')
    plt.close()