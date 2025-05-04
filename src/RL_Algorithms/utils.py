import torch
import torch.nn as nn
import copy
import rospy
from std_msgs.msg import String
import os
import math
import numpy as np
import matplotlib.pyplot as plt

def clone_model(original_model):
    # Create a deep copy of the model
    clone_model = copy.deepcopy(original_model)
    
    # Ensure the clone has the same device as the original
    device = next(original_model.parameters()).device
    clone_model.to(device)

    # Verify that the clone has the same structure and weights
    for (name1, param1), (name2, param2) in zip(original_model.named_parameters(), clone_model.named_parameters()):
      if not torch.all(param1.eq(param2)):
        rospy.logerr(f"Parameters are different !")
    
    
    return clone_model


def target_weight_update( target_network , network , update_coeff):
   
   for target_weight , weight in zip(target_network.parameters(), network.parameters()):
            # Update the weights of network B
            #target_weight.data = update_coeff * weight.data + (1 - update_coeff) * target_weight.data
            target_weight.data.copy_(update_coeff * weight.data + (1 - update_coeff) * target_weight.data)


def init_network_weights(network):
    """
    Initialize weights and biases for any network using Kaiming initialization
    
    Args:
        network: The neural network module to initialize
    """
    def init_layer(layer):
        if isinstance(layer, nn.Linear):
            # Kaiming initialization for weights of linear layers (ReLU activation assumption)
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            # Initialize bias with small positive values
            nn.init.constant_(layer.bias, 0.01)  # Set bias to 0 for better stability
    
    # Apply initialization to all layers
    network.apply(init_layer)
    
    # Special initialization for the final layer
    last_layer = None
    for module in network.modules():
        if isinstance(module, nn.Linear):
            last_layer = module
    
    if last_layer is not None:
        # Initialize final layer with Xavier or Kaiming (try both) for better initial stability
        # Initialize final layer with smaller weights for better initial stability
        nn.init.uniform_(last_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(last_layer.bias, -3e-3, 3e-3)
        # Kaiming initialization for weights of linear layers (ReLU activation assumption)
        #nn.init.kaiming_normal_(last_layer.weight, mode='fan_in', nonlinearity='relu')
        # Initialize bias with small positive values
        #nn.init.constant_(last_layer.bias, 0.01)  # Set bias to 0 for better stability


def score_init_network_weights(network):
    """
    Initialize weights and biases for any network using Kaiming initialization
    
    Args:
        network: The neural network module to initialize
    """
    def init_layer(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # Kaiming for ReLU
            nn.init.constant_(layer.bias, 0)  # Small positive bias
    
    # Apply initialization to all layers
    network.apply(init_layer)



def init_weights(layer, gain=1.0):
    """Orthogonal initialization for linear layers"""
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal(layer.weight.data, gain=gain)
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)  # Small positive bias

def init_layer_norm(layer):
    """Initialize LayerNorm layers"""
    if isinstance(layer, nn.LayerNorm):
        layer.weight.data.fill_(1.0)
        layer.bias.data.fill_(0.0)

def initialize_network(network):
    """Initialize all components of a network"""
    for module in network.modules():
        if isinstance(module, (nn.Linear)):
            # Hidden layers use sqrt(2) gain for ReLU
            init_weights(module, gain=nn.init.calculate_gain('relu'))
        elif isinstance(module, nn.LayerNorm):
            init_layer_norm(module)
    
    # Special initialization for output layers
    #if hasattr(network, 'output_layer'):
    #    init_weights(network.output_layer, gain=1e-3)
        
    return network









            

# need to update
def polar_laser_scan(laser_scan , n_discard_scan , initial_angle , angle_increment):
    start_angle = initial_angle + (n_discard_scan * angle_increment)
    polar_scan = []
    for i in range(len(laser_scan)):
        polar_scan.append(laser_scan[i])
        polar_scan.append(start_angle + (i * (2*angle_increment)))
    return polar_scan

def cartesian_laser_scan(laser_scan , n_discard_scan , initial_angle , angle_increment):
    start_angle = initial_angle + (n_discard_scan * angle_increment)
    cartesian_scan = []
    x_arr = []
    y_arr = []
    for i in range(len(laser_scan)):
        angle = start_angle + (i * (angle_increment))
        x = laser_scan[i] * math.cos(angle)
        x_arr.append(x)
        y = laser_scan[i] * math.sin(angle)
        y_arr.append(y)
        cartesian_scan.append(laser_scan[i] * math.cos(angle))
        cartesian_scan.append(laser_scan[i] * math.sin(angle))
    point_cloud = np.column_stack((np.array(x_arr) , np.array(y_arr)))
    return cartesian_scan , point_cloud

def gen_bounded_scan(laser_scan , max_range = 3.5):            
    for i in range(0 , len(laser_scan)):
        if laser_scan[i] >= max_range:
            laser_scan[i] = max_range
    return laser_scan    


def generate_rays(laser_scan , n_discard_scan , initial_angle  , angle_increment):
    start_angle = initial_angle + (n_discard_scan * angle_increment)
    cartesian_scan = []
    x_arr = []
    y_arr = []
    n = rospy.get_param("/Spatial_Attention/n_rays")
    ray_group_dim = round(len(laser_scan)/n)
    min_rays = 26.0
    min_rays_angle = 0
    for i in range(len(laser_scan)):
        angle = start_angle + (i * angle_increment)
        if laser_scan[i] < min_rays:
            min_rays = laser_scan[i]
            min_rays_angle = angle
        if (i+1) % ray_group_dim == 0:
            if 2*n == len(cartesian_scan):
                break
            #adjust this part for return the cortesian coordinates correct for the thesis
            x = min_rays * math.cos(min_rays_angle)
            x_arr.append(x)
            y = min_rays * math.sin(min_rays_angle)
            y_arr.append(y) 
            cartesian_scan.append(laser_scan[i] * math.cos(angle))
            cartesian_scan.append(laser_scan[i] * math.sin(angle))
            min_rays = 26.0
    if 2*n > len(cartesian_scan) or 2*n < len(cartesian_scan):
        print("Rays array is incorrect ! , n elements : " + str(len(cartesian_scan)))
    if len(cartesian_scan) != 2*n:
        rospy.logerr("error in dimension of input ! " + str(len(input))+ " laser scan dim : " + str(len(laser_scan)))
    pointcloud = np.column_stack((np.array(x_arr) , np.array(y_arr)))
    return cartesian_scan , pointcloud 

def laser_plot(source, target=None, tagd_list=None , rays = False):
    # Plotting the source and target scans
    plt.figure(figsize=(8, 6))

    # Plot source (prev_scan) points in blue
    if rays:
        for i in range( 0,len(source) , 2):
            plt.scatter(source[i], source[i+1], label='Source (Prev Scan)', color='blue', s=5)
    else:    
        plt.scatter(source[:, 0], source[:, 1], label='Source (Prev Scan)', color='blue', s=5)
    if target is not None:
        # Plot target (curr_scan) points in red
        plt.scatter(target[:, 0], target[:, 1], label='Target (Curr Scan)', color='red', s=5)

    if tagd_list is not None:
        # Plot centroids of filtered_prev_scan in green
        for i, data in enumerate(tagd_list):
            if i % 2 == 0:
                plt.scatter(data[0], data[1], c='purple', label='Previous Scan Centroids', s=100)
            else:
                plt.scatter(data[0], data[1], c='orange', label='Current Scan Centroids', s=100)

    # Labeling and formatting
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cartesian Point Cloud with Centroids')
    #plt.legend()
    plt.grid(True)
    plt.show()  





