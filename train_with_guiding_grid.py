#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

'''
This file trains a set of Gaussians, and then applies for 10k iterations the fourier loss w/o any reconstruction loss.
'''

import os
import torch
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from density_grid import SimpleDensityGrid


from torch.utils.tensorboard import SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from torchvision.utils import save_image

def get_plot_array(vis):
    import matplotlib.pyplot as plt
    import io
    import numpy as np
    figure = plt.figure(figsize=(8, 6))  # Adjust the size of the figure
    plt.imshow(vis, cmap='viridis', interpolation='nearest')

    # Add a color bar on the side for reference
    cbar = plt.colorbar()
    cbar.set_label('Probability', rotation=270, labelpad=20)

    # Add labels and title
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)

    plt.tight_layout()  # Adjust layout to fit everything nicely
    buf = io.BytesIO()
    plt.savefig(buf, format='raw')
    buf.seek(0)
    
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    
    w, h = figure.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))[:3,...]
    buf.close()
    plt.close()

    # Add the batch dimension
    return np.expand_dims(im, 0)

def get_fourier_loss(image):
    fourier_image = torch.fft.fftn(image) #rfft2 applies the fourier transform to the last 2 dimensions
    return torch.abs(torch.real(fourier_image)).mean()

def save_image_for_video(gaussians,pipe,background,scene,iteration,opt,test_view_id=75):
    if pipe.save_image_interval>0 and iteration%pipe.save_image_interval==0:
        with torch.no_grad():
            net_image = render(scene.getTestCameras()[test_view_id], gaussians, pipe, background, 1.0)["render"]
            save_image(net_image, f"videos/tmp_frames/frame_{iteration//pipe.save_image_interval - 1}.png")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    sh_degree = 0 if pipe.turn_off_view_dependence else dataset.sh_degree
    gaussians = GaussianModel(sh_degree)
    
    dataset.eval = True
    scene = Scene(dataset, gaussians, shuffle=False)

    # initialize grid
    computing_fourier_loss = opt.grid_size > 0 or opt.adaptive_sampling
    if computing_fourier_loss:
        guiding_grid = SimpleDensityGrid(opt.grid_size,
                                        sample_gmm=opt.number_of_samples>0,
                                        number_of_samples=opt.number_of_samples)

    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print("Guiding Grid is still trained from scratch")
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    ema_fourier_loss = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # if opt.save_image_interval>0: 
    os.makedirs('videos/tmp_frames',exist_ok=True)
    
    for iteration in range(first_iter, opt.iterations + 1):

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Fourier Grid Loss experiment
        
        loss = 0.0
        # if computing_fourier_loss:
        #     if iteration > opt.densify_from_iter + 100:
        #         if opt.adaptive_sampling:
        #             # max_depth_level = min(6, (iteration-600)//100 + 2)
        #             max_depth_level = 6
        #             fourier_loss,real_fourier_loss_tensor = guiding_grid.get_fourier_loss_adaptive(gaussians,max_depth_level=max_depth_level)
        #         else:
        #             fourier_loss,real_fourier_loss_tensor = guiding_grid.get_fourier_loss(gaussians)
        #         real_fourier_loss = real_fourier_loss_tensor.item()
        #         if opt.fourier_loss>0.0:
        #             loss += fourier_loss * opt.fourier_loss
        #     else:
        #         real_fourier_loss = 0.0

        
        # if iteration < opt.apply_fourier_until_iter:
        #     # clip = start_clipping #+ (1.0-start_clipping)/10_000 * (iteration-opt.apply_fourier_from_iter)
        #     clip = 0.5
        #     if iteration > opt.apply_fourier_from_iter and iteration % opt.fourier_interval == 50:
        #         real_fourier_loss = guiding_grid.apply_fourier_loss_n_iterations(gaussians,clip,n_iterations=1000)
                

        # Save Example image
        save_image_for_video(gaussians,pipe,background,scene,iteration,opt)

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if pipe.easy_few_shot>0:
                viewpoint_stack = viewpoint_stack[:pipe.easy_few_shot]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()
        iter_end.record()

        if iteration == opt.iterations:
            progress_bar.close()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
            # ema_fourier_loss = 0.4 * real_fourier_loss + 0.6 * ema_fourier_loss
            
            if iteration % 10 == 0:
                loss_string = f"{ema_loss_for_log:.{7}f}"
                # if computing_fourier_loss: loss_string += f"| Fourier Loss: {ema_fourier_loss:.{7}f}"
                progress_bar.set_postfix({"Loss": loss_string })
                progress_bar.update(10)
            
            losses_to_log = (Ll1, loss, l1_loss)
            if computing_fourier_loss: losses_to_log += (ema_fourier_loss,)

            # Log and save
            training_report(tb_writer, iteration, losses_to_log, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
          
    # Applying Fourier Update after training is finished

    # We only want the Fourier Updates to affect the density field created by the Mixture of Gaussians.
    lr = 0.001  
    l = [
            {'params': [gaussians._xyz], 'lr': lr, "name": "xyz"},
            {'params': [gaussians._opacity], 'lr': lr, "name": "opacity"},
            {'params': [gaussians._scaling], 'lr': lr, "name": "scaling"},
            {'params': [gaussians._rotation], 'lr': lr, "name": "rotation"}
        ]

    fourier_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    # For grid viz
    starting_grid_slice = None
    starting_grid_slice_smoothed = None
    target = None
    
    for iteration_fourier in tqdm(range(10_000)):
        fourier_loss,real_fourier_loss_tensor, total_n_visited_cells, (grid_slice, grid_slice_smoothed), target_output = guiding_grid.get_fourier_loss_adaptive_fixed_target(gaussians,
                                                                                                              clip_value=opt.clip_value,
                                                                                                              max_depth_level=opt.max_depth_level,
                                                                                                              target=target)
        
        if iteration_fourier == 0:
            
            starting_grid_slice = grid_slice.detach()
            
            starting_grid_slice_smoothed = grid_slice_smoothed.detach()# * ratio
            target = target_output.detach() # Replace target just once
            # target = grid_slice_smoothed.detach()# * ratio
            
        if iteration_fourier + opt.iterations in testing_iterations:
            visualization = torch.cat([starting_grid_slice, grid_slice, starting_grid_slice_smoothed],dim=1).clamp(0,0.2).detach().cpu().numpy()
            tb_writer.add_images(f"grid_visualization", get_plot_array(visualization), global_step=iteration_fourier)
        
        #Rendering for loss
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if pipe.easy_few_shot>0:
                viewpoint_stack = viewpoint_stack[:pipe.easy_few_shot]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        with torch.no_grad():
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            
            # Measure mean fourier coefficient of rendered image
            
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        losses_to_log = (Ll1, loss, l1_loss, fourier_loss.item(), total_n_visited_cells)
        training_report(tb_writer, iteration_fourier + opt.iterations, losses_to_log, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
        
        fourier_loss.backward()
        
        fourier_optimizer.step()
        fourier_optimizer.zero_grad(set_to_none = True)
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration,losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if len(losses) == 3:
        Ll1, loss, l1_loss = losses
    elif len(losses) == 4:
        Ll1, loss, l1_loss, fourier_loss = losses
    elif len(losses) == 5:
        Ll1, loss, l1_loss, fourier_loss, total_n_visited_cells = losses
    else:
        raise ValueError(f"Number of losses {len(losses)} is invalid")
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if len(losses) >= 4:
            tb_writer.add_scalar('Mean fourier coefficient',fourier_loss, iteration)
        if len(losses) >= 5:
            tb_writer.add_scalar('Number of Cells Visited',total_n_visited_cells, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                fourier_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    fourier_test += get_fourier_loss(image).double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   
                fourier_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - fourier on image', fourier_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    
    
    test_iterations = [i * 100 + 700 for i in range(0,100)]
    print(test_iterations)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=test_iterations)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    
    fourier_losses = training(lp.extract(args),op.extract(args) , pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # All done
    print("\nTraining complete.")
