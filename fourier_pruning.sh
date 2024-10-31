#!/bin/bash

# Define the scenes and frame rate
main_tensorboard_folder=output/fourier_pruning

# Train the NeRF model on the current scene
experiment_folder_fixed_samples=$main_tensorboard_folder/vanilla
python train_with_fourier_loss.py -s ../data/nerf_synthetic/lego/ \
                            -m $experiment_folder_fixed_samples\
                            --eval\
                            --densify_grad_threshold 0.0008\
                            --easy_few_shot 6 \

depth_level_list=(7 6 5 4 3 2 1)
clip_values=(0.1 0.1 0.1 0.1 0.1 0.1 0.1)

length=${#depth_level_list[@]}
# #0.00016 init original 0.0000016 final original for position lr
for (( i=0; i<$length; i++ )); do
        experiment_folder_fixed_samples=$main_tensorboard_folder/fourier_pruning_${depth_level_list[$i]}
        echo echo "depth level ${depth_level_list[$i]}"

        # Train the NeRF model on the current scene
        python train_with_fourier_loss.py -s ../data/nerf_synthetic/lego/ \
                                        -m $experiment_folder_fixed_samples\
                                        --eval\
                                        --easy_few_shot 6 \
                                        --clip_value ${clip_values[$i]}\
                                        --densify_grad_threshold 0.0008\
                                        --fourier_loss 0\
                                        --max_depth_level ${depth_level_list[$i]}\
                                        --fourier_pruning
done



# Only maximum depth but different maxes # TODO

# for (( i=0; i<$length; i++ )); do
#         experiment_folder_fixed_samples=$main_tensorboard_folder/fourier_pruning_${depth_level_list[$i]}
#         echo echo "depth level ${depth_level_list[$i]}"

#         # Train the NeRF model on the current scene
#         python train_with_fourier_loss.py -s ../data/nerf_synthetic/lego/ \
#                                         -m $experiment_folder_fixed_samples\
#                                         --eval\
#                                         --easy_few_shot 6 \
#                                         --clip_value ${clip_values[$i]}\
#                                         --fourier_loss 0\
#                                         --max_depth_level ${depth_level_list[$i]}\
#                                         --fourier_pruning
# done
