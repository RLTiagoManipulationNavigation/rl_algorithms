import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
from typing import List
import rospy
import rospkg



"""
network_config:
  spatial_key_mlp_layers: [256, 128, 64]        # Embedding network for attention keys
  spatial_value_mlp_layers: [80, 50, 30]        # Feature network for attention values
  spatial_attention_mlp_layers: [60, 50, 1]     # Score network for attention weights
  waypoint_mlp_layers: [64, 40, 30]             # Waypoint embedding MLP (unused in A2)
  action_mlp_layers: [128, 64, 64]              # Output layers for actor/critic

"""


# ================================
# Network Configuration
# ================================
network_config = {
    'spatial_key_mlp_layers': rospy.get_param('/network_config/spatial_key_mlp_layers'),        # Embedding network for attention keys
    'spatial_value_mlp_layers': rospy.get_param('/network_config/spatial_value_mlp_layers'),        # Feature network for attention values
    'spatial_attention_mlp_layers': rospy.get_param('/network_config/spatial_attention_mlp_layers'),     # Score network for attention weights
    'waypoint_mlp_layers': rospy.get_param('/network_config/waypoint_mlp_layers'),             # Waypoint embedding MLP (unused in A2)
    'action_mlp_layers': rospy.get_param('/network_config/action_mlp_layers'),              # Output layers for actor/critic
}


# ================================
# Base Model for Spatial Stream
# ================================
class SpatioTemporalBase(nn.Module):
    """
    Implements the spatial stream of the architecture. Applies location-based attention
    (score+value) to lidar sectors concatenated with waypoints. Omits temporal TAGDs.
    """
    def __init__(self):
        super().__init__()

        # Input dimensionality setup
        self.total_lidar_points = rospy.get_param('/Spatial_Attention/n_rays')                # N lidar points
        self.lidar_point_dim = 2                  # 2D Cartesian input per ray
        #self.robot_state_dim = kwargs["robot_state"]                 # Robot state vector (e.g., velocity)
        self.num_waypoints = rospy.get_param('/Training/n_waypoint')                 # Number of path waypoints
        self.waypoint_dim = 2                 # Dimensionality per waypoint
        self.temporal_seq_len = 1                                   # Temporal window (A2 uses 1)
        self.batch_size = rospy.get_param('/Training/batch_size')  # Batch size for training

        # Sector configuration (Nc sectors)
        self.num_sectors = rospy.get_param('/Spatial_Attention/input_spatial_size')
        self.points_per_sector = self.total_lidar_points // self.num_sectors
        assert self.points_per_sector * self.num_sectors == self.total_lidar_points

        # Input dimension per sector
        '''spatial_input_dim = (
            self.points_per_sector * self.lidar_point_dim +
            self.num_waypoints * self.waypoint_dim +
            self.robot_state_dim
        )'''
        """spatial_input_dim = (
            self.points_per_sector * self.lidar_point_dim +
            self.num_waypoints * self.waypoint_dim 
        )"""

        spatial_input_dim = (
            self.points_per_sector * self.lidar_point_dim +
            self.num_waypoints * 2 
        )

        # Embedding MLP (key)
        self.mlp_spatial_key = build_mlp(
            spatial_input_dim,
            network_config["spatial_key_mlp_layers"],
            activate_last_layer=True
        )

        # Feature MLP (value)
        self.mlp_spatial_value = build_mlp(
            network_config["spatial_key_mlp_layers"][-1] ,
            network_config["spatial_value_mlp_layers"],
            activate_last_layer=False
        )

        # Score MLP (attention weight)
        self.mlp_spatial_attention = build_mlp(
            network_config["spatial_key_mlp_layers"][-1],
            network_config["spatial_attention_mlp_layers"],
            activate_last_layer=False
        )


# ================================
# Actor Network
# ================================
class SpatioTemporalActor(SpatioTemporalBase):
    """
    Actor for A2 ablation. Processes a single lidar scan with spatial attention,
    fuses features, and outputs actions.
    """
    def __init__(self, action_dim ):
        super().__init__()

        # Action output size
        self.num_actions = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output dim: mean + std for SAC, otherwise just action vector
        output_dim = self.num_actions

        # Final MLP projecting aggregated features to action logits
        self.mlp_action_output = build_mlp(
            network_config["spatial_value_mlp_layers"][-1] * self.temporal_seq_len,
            network_config["action_mlp_layers"] + [output_dim],
            activate_last_layer=True , 
            last_layer_activate_func=nn.Tanh()
        )

        #linear and angular velocity raange
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')

        self.debug = rospy.get_param('/Training/debug')
        self.algorithm_name = rospy.get_param("/Training/algorithm")

        self.rospack = rospkg.RosPack()


    def forward(self, lidar_scan , waypoints , eval=False):
        spatial_input = input_split( lidar_scan, waypoints , self.num_sectors)  # Reshape lidar data
        # Compute key, value, attention
        key_features = self.mlp_spatial_key(spatial_input)          # Embedding per sector
        #key_features = torch.cat([key_features, waypoint], dim=2)  # Concatenate waypoints
        value_features = self.mlp_spatial_value(key_features)       # Sector features
        attention_scores = self.mlp_spatial_attention(key_features) # Attention weights

        # Reshape for attention-weighted sum
        value_features = value_features.view(spatial_input.shape[0], self.num_sectors, -1)
        attention_scores = attention_scores.view(spatial_input.shape[0] , self.num_sectors, 1)
       
        #rospy.loginfo(str(attention_weights))
        """if self.debug and eval:
            attention_weights = softmax(attention_scores, dim=1)  # Normalize across sectors
            with open(self.rospack.get_path('tiago_navigation') + "/data/" + str(self.algorithm_name) + "_attention_score.txt", 'a') as file:  
                for i in range(attention_weights.shape[1]):  # Batch dimension
                    file.write(str(attention_weights[0, i, 0].item()))
                    if i < attention_weights.shape[1] - 1:
                        file.write(",")
                file.write("\n")
        """
        # Weighted sum over sectors using softmax attention
        weighted_features, _ = compute_spatial_weighted_feature(attention_scores, value_features)
        #rospy.loginfo("Weighted features shape: %s", str(weighted_features))
        features = weighted_features.view(spatial_input.shape[0], -1)
        # Final action output
        output = self.mlp_action_output(features)
        # Bound the outputs
        #output1 = ((output[:,0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity).unsqueeze(1)
        #output2 = ((output[:,1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity).unsqueeze(1)
          
        #rospy.loginfo(str(output))
        #return torch.cat([output1, output2], dim=1)
        return output

# ================================
# Critic Network
# ================================
class SpatioTemporalCritic(SpatioTemporalBase):
    """
    Critic for A2 ablation. Computes spatial attention over lidar and evaluates Q-value
    of observation-action pair.
    """
    def __init__(self,  action_space ):
        super().__init__()

        # Final Q-value regressor MLP
        self.num_actions = action_space
        self.q_value_mlp = build_mlp(
            network_config["spatial_value_mlp_layers"][-1] * self.temporal_seq_len + self.num_actions,
            network_config["action_mlp_layers"] + [1],
            activate_last_layer=False
        )

    def forward(self, lidar_scan , waypoints , action):

        spatial_input  = input_split( lidar_scan, waypoints , self.num_sectors)  # Reshape lidar data
        # Apply attention block
        key_features = self.mlp_spatial_key(spatial_input)
        #key_features = torch.cat([key_features, waypoint], dim=2)  # Concatenate waypoints
        value_features = self.mlp_spatial_value(key_features)
        attention_scores = self.mlp_spatial_attention(key_features)

        value_features = value_features.view(spatial_input.shape[0], self.num_sectors, -1)
        attention_scores = attention_scores.view(spatial_input.shape[0], self.num_sectors, 1)

        # Attention-based sector aggregation
        weighted_features, _ = compute_spatial_weighted_feature(attention_scores, value_features)
        critic_input = torch.cat([weighted_features.view(spatial_input.shape[0], -1), action], dim=1)
        rospy.loginfo(str(critic_input))
        return self.q_value_mlp(critic_input)


# ================================
# Utilities: MLP and Attention
# ================================
def build_mlp(
    input_dim: int,
    mlp_dims: List[int],
    activate_last_layer: bool = False,
    activate_func=nn.ReLU(),
    last_layer_activate_func=None
) -> nn.Sequential:
    """
    Constructs a feedforward MLP module using a list of dimensions.
    Each layer applies Xavier uniform initialization.

    Args:
        input_dim (int): Dimensionality of the input vector.
        mlp_dims (List[int]): Sizes of subsequent MLP layers.
        activate_last_layer (bool): Whether to activate the last layer.
        activate_func: Activation function to apply between layers (default: ReLU).
        last_layer_activate_func: Optional override for the last layer activation.

    Returns:
        nn.Sequential: Assembled MLP block.
    """
    layers = []
    layer_dims = [input_dim] + mlp_dims
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        torch.nn.init.xavier_uniform_(layers[-1].weight)  # Ensures stable initialization
        nn.init.constant_(layers[-1].bias, 0)
        #torch.nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='relu')

        is_last = (i == len(layer_dims) - 2)
        if not is_last:
            layers.append(activate_func)
        elif activate_last_layer:
            final_act = last_layer_activate_func if last_layer_activate_func else activate_func
            layers.append(final_act)

    return nn.Sequential(*layers)


def compute_spatial_weighted_feature(attention_scores: torch.Tensor, features: torch.Tensor):
    """
    Computes the attention-weighted sum over sectors for a batch.

    Args:
        attention_scores (Tensor): Raw attention logits [B, Nc, 1].
        features (Tensor): Corresponding sector features [B, Nc, D].

    Returns:
        weighted_feature (Tensor): Aggregated feature [B, D].
        attention_weights (Tensor): Normalized attention weights [B, Nc, 1].
    """
    attention_weights = softmax(attention_scores, dim=1)  # Normalize across sectors
    weighted_feature = torch.sum(attention_weights * features, dim=1)  # Weighted sum
    return weighted_feature, attention_weights

def input_split(lidar_data , waypoints , num_sectors):
        # Ensure input has batch dimension
        if len(lidar_data.shape) == 1:
            lidar_data = lidar_data.unsqueeze(0)  # Shape becomes (1, input_dim)
        
        # Ensure waypoints has batch dimension
        if len(waypoints.shape) == 1:
            waypoints = waypoints.unsqueeze(0)  # Shape becomes (1, input_dim) 
        
        lidar_data = lidar_data.reshape(lidar_data.shape[0], num_sectors, -1)
        waypoints = waypoints.view(lidar_data.shape[0], 1, -1).repeat(1, num_sectors, 1)

        # Concatenate sector data and waypoint input
        spatial_input = torch.cat([lidar_data, waypoints], dim=2)
        
        return spatial_input

#######################################
#                                     #
#       DDPG ARCHITECTURE             #
#                                     #
#######################################

class ActorNet(nn.Module):

  def __init__(self, state_dim , action_dim):
    super().__init__()

    #first_layer = rospy.get_param("/Network_param/first_layer")
    #second_layer = rospy.get_param("/Network_param/second_layer")
    
    # Add input normalization 
    #self.ln_input = nn.LayerNorm(state_dim)

    #input layer
    self.mlp_input = nn.Linear( state_dim , 128)
    #self.bn1 = nn.LayerNorm(128)  
    
    #hidden layer
    self.mlp_hid1 = nn.Linear(128, 64)
    #self.bn2 = nn.LayerNorm(64)  
    
    self.mlp_hid2 = nn.Linear(64 , 64)
    #self.bn3 = nn.LayerNorm(64) 
    
    #output layer
    self.mlp_output = nn.Linear( 64 , action_dim)

    #linear and angular velocity raange
    self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
    self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
    self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
    self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')

            # Initialize weights
    self.init_weights()
        
  def init_weights(self):
   
    nn.init.xavier_uniform_(self.mlp_input.weight)
    nn.init.xavier_uniform_(self.mlp_hid1.weight)
    nn.init.xavier_uniform_(self.mlp_hid2.weight)

    nn.init.constant_(self.mlp_input.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.mlp_hid1.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.mlp_hid2.bias, 0.01)  # Small positive bias

        
    # Final layer initialization with smaller weights
    nn.init.uniform_(self.mlp_output.weight, -3e-3, 3e-3)
    nn.init.uniform_(self.mlp_output.bias, -3e-3, 3e-3)


  def forward(self, input):
    # Normalize input
    #input = self.ln_input(input)
    
    a = F.relu(self.mlp_input(input))
    #a = self.bn1(a) 
    
    a = F.relu(self.mlp_hid1(a))
    #a = self.bn2(a)  
    
    a = F.relu(self.mlp_hid2(a))
    #a = self.bn3(a) 
    
    output = torch.tanh(self.mlp_output(a))
    #rospy.loginfo("Actor output before scaling: %s", str(output))
    # Bound the outputs
    output1 = ((output[:,0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity).unsqueeze(1)
    output2 = ((output[:,1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity).unsqueeze(1)
    #output2 = 0.3 * torch.tanh(out[:, 1]).unsqueeze(1)  # Bound to (-3, 3)

    #rospy.loginfo("Actor output after scaling: %s", str(torch.cat([output1, output2], dim=1)))
    return torch.cat([output1, output2], dim=1)

class CriticNet(nn.Module):
   
  def __init__(self, state_dim, action_dim):
    super().__init__()

    #first_layer = rospy.get_param("/Network_param/first_layer")
    #second_layer = rospy.get_param("/Network_param/second_layer")
    
    # Add input normalization 
    #self.ln_input = nn.LayerNorm(state_dim + action_dim)

    #input layer
    self.mlp_input = nn.Linear( state_dim + action_dim, 128)
    #self.bn1 = nn.LayerNorm(128)  
    
    #hidden layer
    self.mlp_hid1 = nn.Linear(128, 64)
    #self.bn2 = nn.LayerNorm(64)  
    
    self.mlp_hid2 = nn.Linear(64 , 64)
    #self.bn3 = nn.LayerNorm(64) 
    
    #output layer
    self.mlp_output = nn.Linear( 64 , 1)

    self.init_weights()
        
  def init_weights(self):
    # Fan-in initialization for linear layers
    nn.init.xavier_uniform_(self.mlp_input.weight)
    nn.init.xavier_uniform_(self.mlp_hid1.weight)
    nn.init.xavier_uniform_(self.mlp_hid2.weight)

    nn.init.constant_(self.mlp_input.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.mlp_hid1.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.mlp_hid2.bias, 0.01)  # Small positive bias

    # Final layer initialization with smaller weights
    nn.init.uniform_(self.mlp_output.weight, -3e-4, 3e-4)
    nn.init.uniform_(self.mlp_output.bias, -3e-4, 3e-4)

  def forward(self, state, action):
    input = torch.cat([state, action], dim=1)
    
    # Normalize combined input
    #input = self.ln_input(input)
    
    a = F.relu(self.mlp_input(input))
    #a = self.bn1(a) 
    
    a = F.relu(self.mlp_hid1(a))
    #a = self.bn2(a) 
    
    a = F.relu(self.mlp_hid2(a))
    #a = self.bn3(a) 

    return self.mlp_output(a)
    

#######################################
#                                     #
#       ATTENTION ARCHITECTURE        #
#                                     #
#######################################    

#Embedding Network
class Embedding(nn.Module):
  def __init__(self, input_size):
    super().__init__()

    self.input_size = input_size 
    #self.ln_input = nn.LayerNorm(input_size)
    
    #input layer
    self.mlp_input = nn.Linear(input_size, 256)
    #self.bn1 = nn.LayerNorm(256)  
    
    #hidden layer
    self.mlp_hid = nn.Linear(256, 128)
    #self.bn2 = nn.LayerNorm(128)  
    
    #output layer
    self.mlp_output = nn.Linear(128, 64)
    # No normalization needed at output as other networks expect raw embeddings
    
  def forward(self, input):
    #input = self.ln_input(input)
    
    a = F.relu(self.mlp_input(input))
    #a = self.bn1(a) 
    
    a = F.relu(self.mlp_hid(a))
    #a = self.bn2(a)  
    
    return F.relu(self.mlp_output(a))

  
#Feature Network
class Feature(nn.Module):
  def __init__(self, input_size):
    super().__init__()

    self.input_size = input_size 
    #self.ln_input = nn.LayerNorm(input_size)
    
    #input layer
    self.mlp_input = nn.Linear(input_size, 80)
    #self.bn1 = nn.LayerNorm(80) 
    
    #hidden layer
    self.mlp_hid = nn.Linear(80, 50)
    #self.bn2 = nn.LayerNorm(50)  
    
    #output layer
    self.mlp_output = nn.Linear(50, 30)
    # No normalization at output to preserve feature diversity

  def forward(self, input):
    #input = self.ln_input(input)
    
    a = F.relu(self.mlp_input(input))
    #a = self.bn1(a) 
    
    a = F.relu(self.mlp_hid(a))
    #a = self.bn2(a)  
    
    return self.mlp_output(a)
  
#Score Network
class Score(nn.Module):
  def __init__(self, input_size):
    super().__init__()

    self.input_size = input_size 
    #self.ln_input = nn.LayerNorm(input_size)
    
    #input layer
    self.mlp_input = nn.Linear(input_size, 60)
    #self.ln1 = nn.LayerNorm(80) 
    
    #hidden layer
    self.mlp_hid = nn.Linear(60, 50)
    #self.ln2 = nn.LayerNorm(50)  
    
    #output layer
    self.mlp_output = nn.Linear(50, 1)
    # No normalization at output as we want raw scores

  def forward(self, input):
    #input = self.ln_input(input)
    
    a = F.relu(self.mlp_input(input))
    #a = self.ln1(a)  
    
    a = F.relu(self.mlp_hid(a))
    #a = self.ln2(a) 
    
    return self.mlp_output(a)
  
class MLP(nn.Module):
    def __init__(self, input_size, output_size, final_layer=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, self.output_size)
        
        # Add normalization for non-final layers
        if not final_layer:
            self.norm = nn.LayerNorm(self.output_size)
        else:
            self.norm = None

    def forward(self, input):
        output = F.relu(self.fc(input))
        if self.norm is not None:
            output = self.norm(output)
        return output