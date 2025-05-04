import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy
from RL_Algorithms.replay_buffer import ReplayBuffer , Replay_Buffer
from RL_Algorithms.attention_module import Spatial_Attention
from RL_Algorithms.temporal_attention import Temporal_Attention
import rospy
from RL_Algorithms.OUNoise import OUNoise , GaussianNoise
from RL_Algorithms.utils import *
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
from torch.distributions import Normal
import random

def init_weights(m, gain=1.0):
    """
    Orthogonal initialization for the weights and zero initialization for the biases
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0.0)

# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        first_layer = rospy.get_param("/Network_param/first_layer")
        second_layer = rospy.get_param("/Network_param/second_layer")
        rospy.loginfo(str(first_layer) + " " + str(second_layer) + " " + str(state_dim) + " " + str(action_dim))
        #self.ln_input = nn.LayerNorm(state_dim) 
        self.l1 = nn.Linear(state_dim, first_layer)
        #self.bn1 = nn.LayerNorm(128)
        self.l2 = nn.Linear(first_layer, second_layer)
        #self.bn2 = nn.LayerNorm(64)
        #self.l3 = nn.Linear(64, action_dim)
        
        # Mean and log_std outputs
        self.mu_layer = nn.Linear( second_layer, action_dim)
        self.log_std_layer = nn.Linear( second_layer , action_dim)

        # Initialize weights
        self.apply(init_weights)
        
        # Special initialization for output layers
        # Mean layer: smaller weights for stable initial outputs
        nn.init.orthogonal_(self.mu_layer.weight.data, gain=0.01)
        nn.init.constant_(self.mu_layer.bias.data, 0.0)
        
        # Log_std layer: initialize to small negative values for reasonable initial exploration
        nn.init.orthogonal_(self.log_std_layer.weight.data, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias.data, -1.0)  # Start with moderate exploration
        
        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max
        
    def forward(self, state, deterministic):
        '''Network with Enforcing Action Bounds'''
		#state = self.ln_input(state)
        net_out = F.relu(self.l1(state))
		#a = self.bn1(a)  # Add normalization after ReLU
        net_out = F.relu(self.l2(net_out))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX) 
		# we learn log_std rather than std, so that exp(log_std) is always > 0
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic: u = mu
        else: u = dist.rsample()
        
        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        action = torch.tanh(u)
		# Get probability density of logp_pi_a from probability density of u:
		# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
		# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
        # Enforcing action bounds (from SpinningUp implementation)
        logp_pi = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        return action, logp_pi
    
# Critic Network (Q-Value)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        first_layer = rospy.get_param("/Network_param/first_layer")
        second_layer = rospy.get_param("/Network_param/second_layer")
        #self.ln_input = nn.LayerNorm(state_dim + action_dim)
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, first_layer)
        #self.bn1 = nn.LayerNorm(128)
        self.l2 = nn.Linear(first_layer , second_layer)
        #self.bn2 = nn.LayerNorm(64)
        self.l3 = nn.Linear(second_layer , 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, first_layer)
        #self.bn4 = nn.LayerNorm(128)
        self.l5 = nn.Linear(first_layer , second_layer)
        #self.bn5 = nn.LayerNorm(64)
        self.l6 = nn.Linear(second_layer , 1)

        # Initialize weights
        self.apply(init_weights)
        
        # Special initialization for output layers
        # Slightly smaller weights for Q output layers
        nn.init.orthogonal_(self.l3.weight.data, gain=0.1)
        nn.init.orthogonal_(self.l6.weight.data, gain=0.1)
        nn.init.constant_(self.l3.bias.data, 0.0)
        nn.init.constant_(self.l6.bias.data, 0.0)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.l1(sa))
        #q1 = self.bn1(q1)
        q1 = F.relu(self.l2(q1))
        #q1 = self.bn2(q1)
        q1 = self.l3(q1)
        
        # Q2
        q2 = F.relu(self.l4(sa))
        #q2 = self.bn4(q2)
        q2 = F.relu(self.l5(q2))
        #q2 = self.bn5(q2)
        q2 = self.l6(q2)
        
        return q1, q2

class SAC(nn.Module):
    def __init__(
            self, 
            env
        ):
        super().__init__()
		#Setting environment param
        self.state_dim = env.observation_space.shape[0]        
        self.action_dim = env.action_space.shape[0]

		#Set the structure of the model used 
        self.spatial_flag = rospy.get_param("/Architecture_modifier/spatial_att")
        self.temporal_flag = rospy.get_param("/Architecture_modifier/temporal_att")
        self.att_flag = rospy.get_param("/Training/attention_module_flag")
        self.adaptive_entropy = rospy.get_param("/SAC/adaptive_entropy")
        self.att_flag = rospy.get_param("/Training/attention_module_flag")
        
        self.ddpg_input_dim = rospy.get_param("/Spatial_Attention/spatial_att_ourdim")
        self.spatial_input_size = rospy.get_param("/Spatial_Attention/n_sector_spatialatt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.spatial_attention = Spatial_Attention(2*self.spatial_input_size + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)  
		
		# Add output normalization for attention to stabilize TD3 input
		#self.spatial_output_norm = nn.LayerNorm(30).to(self.device)
		#self.no_att_norm = nn.LayerNorm(632).to(self.device)

		#linear and angular velocity raange
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')

		#value for generate the correct version of laser scan 
        self.initial_angle = rospy.get_param("/Tiago/initial_angle")
        self.angle_increment = rospy.get_param("/Tiago/angle_increment")
        self.n_discard_scan = rospy.get_param("/Tiago/remove_scan")

        # Initialize networks
        if self.att_flag:
            self.actor = Actor(30 , self.action_dim).to(self.device)
            self.critic = Critic(30 , self.action_dim).to(self.device)
        else:
			#self.actor = Actor(632 , self.action_dim).to(self.device)
			#self.critic = Critic(632 , self.action_dim).to(self.device)
            self.actor = Actor(90 + 10 , self.action_dim).to(self.device)
            self.critic = Critic(90 + 10 , self.action_dim).to(self.device)
		# Initialize network weights
        #init_network_weights(self.actor)
        #init_network_weights(self.critic)

        # Create target networks
        #self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Setup optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)		
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.spatial_attention_optimizer = torch.optim.Adam(self.spatial_attention.parameters(), lr=3e-4)
        
        self.replay_buffer = Replay_Buffer(rospy.get_param("/Training/buffer_size"))

        # Initialize noise processes for exploration
        self.exploration_noise = OUNoise(self.action_dim)
        self.gaussian_noise = GaussianNoise(self.action_dim)
        
        # Hyperparameters
        self.batch_size = rospy.get_param("/Training/batch_size")
        self.tau = 0.005
        # Entropy tuning
        if self.adaptive_entropy:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 0.12
        self.gamma = 0.99
    
    def forward(self, spatial_input, waypoints, goal_pos, add_noise=True):
        # Convert inputs to proper tensor format
        spatial_input = torch.tensor(spatial_input, dtype=torch.float32).to(self.device)
        waypoints = torch.tensor(waypoints, dtype=torch.float32).to(self.device)
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        if self.att_flag:
            if add_noise:
                # Get spatial attention output
                spatial_out = self.spatial_attention(spatial_input, waypoints)
            else:
                spatial_out = self.spatial_attention(spatial_input, waypoints , True)
        else:
			# Ensure input has batch dimension
            if len(spatial_input.shape) == 1:
                spatial_input = spatial_input.unsqueeze(0)  # Shape becomes (1, input_dim)
        
        	# Ensure waypoints has batch dimension
            if len(goal_pos.shape) == 1:
                goal_pos = goal_pos.unsqueeze(0)  # Shape becomes (1, input_dim) 
                
            if len(waypoints.shape) == 1:
                waypoints = waypoints.unsqueeze(0) 
			#norm_input = self.no_att_norm(torch.cat((spatial_input, goal_pos), dim=1))
			#norm_input = torch.cat((spatial_input, goal_pos), dim=1)
            spatial_out = torch.cat((spatial_input, waypoints), dim=1)        
		# Apply normalization to stabilize the TD3 input
		#normalized_spatial_out = self.spatial_output_norm(spatial_out)
        #rospy.loginfo(spatial_out.size())
		# Get raw action from actor
        if add_noise:
            action, _ = self.actor(spatial_out , deterministic = False)
            
            #rospy.loginfo(str(action[0]))
        else:
            action, _ = self.actor(spatial_out , deterministic = True)
        action = action[0].cpu()
        # Scale action to actual velocity range
        scaled_action = torch.zeros(2)
        scaled_action[0] = (action[0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity
        scaled_action[1] = (action[1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity
        
        # Clamp action to allowed range
        scaled_action[0] = torch.clamp(scaled_action[0], min=self.min_linear_velocity, max=self.max_linear_velocity)
        scaled_action[1] = torch.clamp(scaled_action[1], min=-self.max_angular_velocity, max=self.max_angular_velocity)
        #rospy.loginfo(str(scaled_action))
        return scaled_action
    
    def update(self , path):
        if int(self.replay_buffer.count()) < 500:
            return
        # Sample a batch from the replay buffer
        state, action, next_state, reward, waypoints, next_waypoints, extra_input, next_extra_input , is_done = self.replay_buffer.sample(self.batch_size)
        
        # Sample replay buffer
        #state, action, next_state, reward, waypoints, next_waypoints, extra_input, next_extra_input = self.replay_buffer.sample(self.batch_size)
        if self.att_flag:
            # Get spatial attention outputs
            curr_spatial_out = self.spatial_attention(state, waypoints)
            next_spatial_out = self.spatial_attention(next_state, next_waypoints)
        else:
            curr_spatial_out = torch.cat((state, waypoints), dim=1)
            next_spatial_out = torch.cat((next_state, next_waypoints), dim=1)        
        ##### UPDATE CRITIC #####

        with torch.no_grad():
            # Sample next action and log_prob from target policy
            next_action, next_log_pi = self.actor(next_spatial_out , deterministic=False ) 
            
            # Compute target Q values
            target_Q1, target_Q2 = self.critic_target(next_spatial_out, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Compute soft Q target with entropy term
            target_Q = reward + is_done * self.gamma * (target_Q - self.alpha * next_log_pi)
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(curr_spatial_out, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Log metrics
        with open(path + "/critic_loss.txt", 'a') as file:
            file.write(str(critic_loss.item()) + "\n")
            
        with open(path + "/critic_diff.txt", 'a') as file:
            value_diff = torch.abs(current_Q1 - current_Q2)
            mean_diff = torch.mean(value_diff).item()
            file.write(str(mean_diff) + "\n")
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        ##### UPDATE ACTOR #####

		# Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.critic.parameters(): params.requires_grad = False
        
        # Compute policy loss
        action , log_pi = self.actor(curr_spatial_out , deterministic=False)
        current_Q1 , current_Q2 = self.critic(curr_spatial_out, action)
        Q = torch.min(current_Q1, current_Q2)
        
        # Policy loss is expectation of Q - entropy
        policy_loss = (self.alpha * log_pi - Q).mean()
        
        # Log policy loss
        with open(path + "/actor_loss.txt", 'a') as file:
            file.write(str(policy_loss.item()) + "\n")
        
        # Optimize policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_optimizer.step()

        # Spatial attention update
        self.spatial_attention_optimizer.zero_grad()
        # A specific loss for spatial attention or recompute actor_loss if needed
        spatial_loss = (self.alpha * log_pi - Q).mean()
        spatial_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.spatial_attention.parameters(), max_norm=5.0)
        self.spatial_attention_optimizer.step()

        for params in self.critic.parameters(): params.requires_grad = True

        ##### UPDATE ALPHA PARAM #####
        
        if self.adaptive_entropy:
            # Optimize temperature parameter alpha
            alpha_loss = -self.log_alpha * (log_pi.detach() + self.target_entropy).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
        # Logging
        '''with open(path + "/alpha.txt", 'a') as file:
            file.write(str(self.alpha.item()) + "\n")
            
        with open(path + "/entropy.txt", 'a') as file:
            file.write(str(-log_pi.mean().item()) + "\n")'''
        
        # Log gradient information
        rospy.loginfo(self.check_gradients(self.spatial_attention))
        rospy.loginfo(self.check_gradients(self.actor))
        rospy.loginfo(self.check_gradients(self.critic))
        
        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Diagnostic function
    def check_gradients(self, model):
        result = str(model) + "\n"
        for name, param in model.named_parameters():
            if param.grad is not None:
                result += f"{name}: grad norm = {param.grad.norm().item()}\n"
            else:
                result += f"{name}: No gradient\n"
        return result
    
    def save(self, filename, folder_path):
        # Save all model components
        try:
            os.makedirs(folder_path, exist_ok=True)
            torch.save(self.critic.state_dict(), os.path.join(folder_path, filename + "_critic"))
            torch.save(self.critic_optimizer.state_dict(), os.path.join(folder_path, filename + "_critic_optimizer"))
            
            torch.save(self.actor.state_dict(), os.path.join(folder_path, filename + "_actor"))
            torch.save(self.actor_optimizer.state_dict(), os.path.join(folder_path, filename + "_actor_optimizer"))
            
            torch.save(self.spatial_attention.state_dict(), os.path.join(folder_path, filename + "_spatial_attention"))
            torch.save(self.spatial_attention_optimizer.state_dict(), os.path.join(folder_path, filename + "_spatialatt_optimizer"))
        
			# Save the new output normalization layer
			#torch.save(self.spatial_output_norm.state_dict(), os.path.join(folder_path, filename + "_spatial_output_norm"))
            rospy.loginfo(f"Model successfully saved to {folder_path}")
        except Exception as e:
            rospy.logerr(f"Error saving model: {str(e)}")
    
    def load(self, filename, folder_path):
        """Load all model components."""
        try:
            # Load critic
            self.critic.load_state_dict(torch.load(os.path.join(folder_path, filename + "_critic")))
            self.critic_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_critic_optimizer")))
            self.critic_target.load_state_dict(torch.load(os.path.join(folder_path, filename + "_critic_target")))
            
            # Load policy
            self.policy.load_state_dict(torch.load(os.path.join(folder_path, filename + "_policy")))
            self.policy_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_policy_optimizer")))
            
            # Load spatial attention
            self.spatial_attention.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatial_attention")))
            self.spatial_attention_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatialatt_optimizer")))
            
            # Load temperature parameter
            self.log_alpha = torch.load(os.path.join(folder_path, filename + "_log_alpha"))
            self.alpha_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_alpha_optimizer")))
            self.alpha = self.log_alpha.exp()
            
            rospy.loginfo(f"Model successfully loaded from {folder_path}")
        except Exception as e:
            rospy.logerr(f"Error loading model: {str(e)}")
            
    def update_buffer(self, state, new_state, reward, action, waypoints, next_waypoints, extra_input, next_extra_input , is_done):
        """Add experience to replay buffer."""
        try:
            self.replay_buffer.add(state, action, new_state, reward, waypoints, next_waypoints, extra_input, next_extra_input , is_done)
        except Exception as e:
            rospy.logerr(f"Error updating buffer: {str(e)}")
