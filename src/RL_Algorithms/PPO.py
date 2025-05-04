import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy
from RL_Algorithms.replay_buffer import ReplayBuffer
from RL_Algorithms.attention_module import Spatial_Attention
from RL_Algorithms.temporal_attention import Temporal_Attention
from RL_Algorithms.utils import *
import torch.optim as optim
import os
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Proximal Policy Optimization (PPO)
# Paper: https://arxiv.org/abs/1707.06347


class ActorNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(ActorNetwork, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
        #mu = torch.tanh(self.mu_head(a))
		mu = torch.sigmoid(self.mu_head(a))
        
        #return mu

	def get_dist(self,state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist

	def deterministic_act(self, state):
		return self.forward(state)


class CriticNetwork(nn.Module):
	def __init__(self, state_dim,net_width):
		super(CriticNetwork, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v


class PPO(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Setting environment param
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Set the structure of the model used 
        self.spatial_flag = rospy.get_param("/Architecture_modifier/spatial_att")
        self.temporal_flag = rospy.get_param("/Architecture_modifier/temporal_att")
        self.att_flag = rospy.get_param("/Training/attention_module_flag")
        
        self.ddpg_input_dim = rospy.get_param("/Spatial_Attention/spatial_att_ourdim")
        self.spatial_input_size = rospy.get_param("/Spatial_Attention/n_sector_spatialatt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.spatial_attention = Spatial_Attention(2*self.spatial_input_size + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)
        self.temporal_attention = Temporal_Attention(4 + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)
        
        # Linear and angular velocity range
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')
        
        # Values for laser scan
        self.initial_angle = rospy.get_param("/Tiago/initial_angle")
        self.angle_increment = rospy.get_param("/Tiago/angle_increment")
        self.n_discard_scan = rospy.get_param("/Tiago/remove_scan")
        
        self.actor = ActorNetwork(30 , self.action_dim).to(self.device)
        self.critic = CriticNetwork(30 , self.action_dim).to(self.device)
        
        
        # Initialize network weights
        init_network_weights(self.actor)
        init_network_weights(self.critic)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.spatial_attention_optimizer = torch.optim.Adam(self.spatial_attention.parameters(), lr=1e-4)
        
        # PPO hyperparameters
        self.ppo_epochs = rospy.get_param("/PPO/ppo_epochs", 10)
        self.clip_param = rospy.get_param("/PPO/clip_param", 0.2)
        self.value_loss_coef = rospy.get_param("/PPO/value_loss_coef", 0.5)
        self.entropy_coef = rospy.get_param("/PPO/entropy_coef", 0.01)
        self.max_grad_norm = rospy.get_param("/PPO/max_grad_norm", 0.5)
        self.batch_size = rospy.get_param("/PPO/batch_size", 64)
        self.gamma = rospy.get_param("/PPO/gamma", 0.99)
        self.gae_lambda = rospy.get_param("/PPO/gae_lambda", 0.95)
        
        # Store training data
        self.states = []
        self.actions = []
        self.action_means = []
        self.log_stds = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.waypoints = []
        
    def forward(self, spatial_input, waypoints, goal_pos, deterministic=False):
        # Convert inputs to proper tensor format
        spatial_input = torch.tensor(spatial_input, dtype=torch.float32).to(self.device)
        waypoints = torch.tensor(waypoints, dtype=torch.float32).to(self.device)
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        if self.att_flag:
            # Get spatial attention output
            spatial_out = self.spatial_attention(spatial_input, waypoints)
            
            # Ensure spatial_out has batch dimension
            if len(spatial_out.shape) == 1:
                spatial_out = spatial_out.unsqueeze(0)
                
            # Get action from actor_critic network
            action, raw_action, action_mean, log_std = self.actor_critic.get_action(spatial_out, deterministic)
        else:
            # Ensure input has batch dimension
            if len(spatial_input.shape) == 1:
                spatial_input = spatial_input.unsqueeze(0)
                
            # Ensure waypoints has batch dimension
            if len(waypoints.shape) == 1:
                waypoints = waypoints.unsqueeze(0)
                
            combined_input = torch.cat((spatial_input, waypoints), dim=1)
            
            # Get action from actor_critic network
            action, raw_action, action_mean, log_std = self.actor_critic.get_action(combined_input, deterministic)
        
        # Return scaled and clamped action
        return action.cpu().detach().numpy()[0], raw_action.cpu().detach(), action_mean.cpu().detach(), log_std.cpu().detach()
    
    def store(self, state, action, action_mean, log_std, reward, value, log_prob, done, waypoint):
        # Store trajectory step
        self.states.append(state)
        self.actions.append(action)
        self.action_means.append(action_mean)
        self.log_stds.append(log_std)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.waypoints.append(waypoint)
    
    def compute_gae(self, next_value, gamma, gae_lambda):
        values = self.values + [next_value]
        gae = 0
        returns = []
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])
            
        return returns
    
    def update(self, path, next_value=0):
        # If not enough data, skip update
        if len(self.states) < self.batch_size:
            return
        
        '''Prepare PyTorch data from Numpy data'''
        state = torch.from_numpy(self.s_hoder).to(self.dvc)
        action = torch.from_numpy(self.a_hoder).to(self.dvc)
        reward = torch.from_numpy(self.r_hoder).to(self.dvc)
        next_state = torch.from_numpy(self.s_next_hoder).to(self.dvc)
        waypoints = torch.from_numpy(self.waypoints_hoder).to(self.dvc)
        next_waypoints = torch.from_numpy(self.next_waypoints_hoder).to(self.dvc)
        logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
        done = torch.from_numpy(self.done_hoder).to(self.dvc)
        dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

		# Get spatial attention outputs
        curr_spatial_out = self.spatial_attention(state, waypoints)
        next_spatial_out = self.spatial_attention(next_state, next_waypoints)

        
            
        
                
    def clear_trajectory(self):
        # Reset all trajectory data
        self.states = []
        self.actions = []
        self.action_means = []
        self.log_stds = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.waypoints = []
        
    # Diagnostic function
    def check_gradients(self, model):
        rospy.logwarn(str(model))
        for name, param in model.named_parameters():
            if param.grad is not None:
                rospy.logwarn(f"{name}: grad norm = {param.grad.norm().item()}")
            else:
                rospy.logwarn(f"{name}: No gradient")
                
    def save(self, filename, folder_path):
        # Save all model components
        try:
            os.makedirs(folder_path, exist_ok=True)
            torch.save(self.actor_critic.state_dict(), os.path.join(folder_path, filename + "_actor_critic"))
            torch.save(self.optimizer.state_dict(), os.path.join(folder_path, filename + "_optimizer"))
            torch.save(self.spatial_attention.state_dict(), os.path.join(folder_path, filename + "_spatial_attention"))
            torch.save(self.spatial_attention_optimizer.state_dict(), os.path.join(folder_path, filename + "_spatialatt_optimizer"))
            
            rospy.loginfo(f"Model successfully saved to {folder_path}")
        except Exception as e:
            rospy.logerr(f"Error saving model: {str(e)}")
            
    def load(self, filename, folder_path):
        # Load all model components
        try:
            # Load actor_critic
            self.actor_critic.load_state_dict(torch.load(os.path.join(folder_path, filename + "_actor_critic")))
            self.optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_optimizer")))
            
            # Load spatial attention
            self.spatial_attention.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatial_attention")))
            self.spatial_attention_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatialatt_optimizer")))
            
            rospy.loginfo(f"Model successfully loaded from {folder_path}")
        except Exception as e:
            rospy.logerr(f"Error loading model: {str(e)}")
            
    def process_state(self, state, waypoints, goal_pos):
        """
        Process state and get action value
        """
        # Convert inputs to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        waypoints_tensor = torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if self.att_flag:
            # Get spatial attention output
            spatial_out = self.spatial_attention(state_tensor, waypoints_tensor)
            
            # Get action from actor_critic network
            _, _, value = self.actor_critic(spatial_out)
        else:
            # Without attention
            combined_input = torch.cat((state_tensor, waypoints_tensor), dim=1)
            
            # Get action from actor_critic network
            _, _, value = self.actor_critic(combined_input)
            
        return value
        
    def collect_experience(self, state, action, action_mean, log_std, reward, done, waypoint):
        """
        Collect experience for PPO training
        """
        # Convert state to tensor and process
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        waypoint_tensor = torch.tensor(waypoint, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        if self.att_flag:
            # Get spatial attention output
            spatial_out = self.spatial_attention(state_tensor, waypoint_tensor)
            
            # Evaluate the action to get log probability and value
            log_prob, _, value = self.actor_critic.evaluate_actions(spatial_out, action_tensor.to(self.device))
        else:
            # Without attention
            combined_input = torch.cat((state_tensor, waypoint_tensor), dim=1)
            
            # Evaluate the action to get log probability and value
            log_prob, _, value = self.actor_critic.evaluate_actions(combined_input, action_tensor.to(self.device))
            
        # Store the experience
        self.store(
            state_tensor, 
            action_tensor, 
            torch.tensor(action_mean, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(log_std, dtype=torch.float32).unsqueeze(0),
            reward, 
            value, 
            log_prob.unsqueeze(0), 
            done,
            waypoint_tensor
        )
