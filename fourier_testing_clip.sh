#!/bin/bash

# Define the scenes and frame rate
main_tensorboard_folder=output/fourier_loss_test_4

fourier_weights=(10.0 1.0 0.1 0.01 0)
depth_level_list=(7 6 5 2)
clip_values=(0.01 0.05 0.1 0.1)

length=${#depth_level_list[@]}
# #0.00016 init original 0.0000016 final original for position lr
for (( i=0; i<$length; i++ )); do
    for fw in "${fourier_weights[@]}"
        do
            experiment_folder_fixed_samples=$main_tensorboard_folder/experiment_dl_${depth_level_list[$i]}_cv_${clip_values[$i]}_fw_$fw
            echo echo "depth level ${depth_level_list[$i]} with clip value ${clip_values[$i]} and fourier weight $fw"

            # Train the NeRF model on the current scene
            python train_with_fourier_loss.py -s ../data/nerf_synthetic/lego/ \
                                            -m $experiment_folder_fixed_samples\
                                            --eval\
                                            --iterations 2000\
                                            --densify_grad_threshold 0.0008\
                                            --densify_from_iter 600\
                                            --easy_few_shot 6 \
                                            --clip_value ${clip_values[$i]}\
                                            --fourier_loss $fw\
                                            --max_depth_level ${depth_level_list[$i]}
                                            
        done
done

length=${#depth_level_list[@]}
# #0.00016 init original 0.0000016 final original for position lr
for (( i=0; i<$length; i++ )); do
    for fw in "${fourier_weights[@]}"
        do
            experiment_folder_fixed_samples=$main_tensorboard_folder/experiment_with_fourier_pruning_dl_${depth_level_list[$i]}_cv_${clip_values[$i]}_fw_$fw
            echo echo "depth level ${depth_level_list[$i]} with clip value ${clip_values[$i]} and fourier weight $fw"

            # Train the NeRF model on the current scene
            python train_with_fourier_loss.py -s ../data/nerf_synthetic/lego/ \
                                            -m $experiment_folder_fixed_samples\
                                            --eval\
                                            --iterations 2000\
                                            --densify_grad_threshold 0.0008\
                                            --densify_from_iter 600\
                                            --easy_few_shot 6 \
                                            --clip_value ${clip_values[$i]}\
                                            --fourier_loss $fw\
                                            --max_depth_level ${depth_level_list[$i]}\
                                            --fourier_pruning

        done
done
