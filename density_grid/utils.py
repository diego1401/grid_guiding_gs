from torch.distributions import constraints
import torch 
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical


def create_sphere_array(grid_size, r_ratio):
    """
    Create an nxm array with a circle of radius r marked with 1.0 inside and 0.0 outside.
    
    Parameters:
    n (int): Number of rows.
    m (int): Number of columns.
    r_ratio (float): Ratio of the radius of the circle to the smaller dimension of the array.
    
    Returns:
    numpy.ndarray: The resulting array.
    """
    n,m,k = grid_size,grid_size,grid_size
    # Calculate the radius
    cube_side=1 # TODO: should be the max
    diag = np.sqrt(3*cube_side**2)
    radius=r_ratio * diag
    # Create an nxm array of zeros
    if r_ratio >= 1.0:
        array = torch.ones((n, m,k))
        return array
    array = torch.zeros((n,m,k))
    
    x_coords = y_coords = z_coords = torch.linspace(-1, 1, grid_size)
    grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords,z_coords, indexing='xy')  # grid_x and grid_y of shape (H,W,D)

    # Stack to get coordinates tensor of shape (H,W,2)
    # coords = torch.stack([grid_x, grid_y,grid_z], dim=-1) # (H,W,D,3)
    
    # Calculate the distance of each point from the center
    distance_from_center = np.sqrt((grid_x)**2 + (grid_y)**2 + (grid_z)**2)
    
    # Set the values within the radius to 1.0
    array[distance_from_center <= radius] = 1.0
    return array

def create_distance_sphere(grid_size):
    """
    Create an nxm array with a circle of radius r marked with 1.0 inside and 0.0 outside.
    
    Parameters:
    n (int): Number of rows.
    m (int): Number of columns.
    r_ratio (float): Ratio of the radius of the circle to the smaller dimension of the array.
    
    Returns:
    numpy.ndarray: The resulting array.
    """
    n,m,k = grid_size,grid_size,grid_size
    # Calculate the radius
    cube_side=1 # TODO: should be the max
    diag = np.sqrt(3*cube_side**2)
    
    array = torch.zeros((n,m,k))
    
    x_coords = y_coords = z_coords = torch.linspace(-1, 1, grid_size)
    grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords,z_coords, indexing='xy')  # grid_x and grid_y of shape (H,W,D)

    # Stack to get coordinates tensor of shape (H,W,2)
    # coords = torch.stack([grid_x, grid_y,grid_z], dim=-1) # (H,W,D,3)
    
    # Calculate the distance of each point from the center
    distance_from_center = np.sqrt((grid_x)**2 + (grid_y)**2 + (grid_z)**2)
    
    return distance_from_center

def check_gmm_constraints(gaussians):
    points_to_prune = constraints.positive_definite.check(gaussians.get_covariance_matrix())
    return points_to_prune
    # if (points_to_prune==False).any():
    #     n_gaussians = gaussians.get_xyz.shape[0]
    #     gaussians.prune_points(~points_to_prune)
    #     n_pruned_points = (~points_to_prune).sum().item()
    #     print(f"Pruned {n_pruned_points} gaussians")
    #     if n_gaussians == n_pruned_points:
    #         raise ValueError("Pruned all the gaussians.")
    #     return False
    # return True


def define_gmm(mu,cov,alpha):
    comp = MultivariateNormal(mu, cov)
    
    mix = Categorical(alpha.squeeze(),validate_args = False) # We are sure opacities when normalized give valid probability distributions
    gmm = MixtureSameFamily(mix, comp)
    return gmm