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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()
		"""self.ln_input = nn.LayerNorm(state_dim) 
		self.l1 = nn.Linear(state_dim, 256)
		#self.bn1 = nn.LayerNorm(128)
		self.l2 = nn.Linear(256, 128)
		#self.bn2 = nn.LayerNorm(64)
		self.l3 = nn.Linear(128, action_dim)"""

		#self.ln_input = nn.LayerNorm(state_dim) 
		self.l1 = nn.Linear(state_dim, 256)
		#self.bn1 = nn.LayerNorm(128)
		self.l2 = nn.Linear(256, 128)
		#self.bn2 = nn.LayerNorm(64)
		self.l3 = nn.Linear(128, action_dim)
		
		self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
		self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
		self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
		self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')
		

	def forward(self, state):
		#state = self.ln_input(state)
		a = F.relu(self.l1(state))
		#a = self.bn1(a)  # Add normalization after ReLU
		a = F.relu(self.l2(a))
		#a = self.bn2(a)  # Add normalization after ReLU
		output = torch.tanh(self.l3(a))
		# Bound the outputs
		output1 = ((output[:,0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity).unsqueeze(1)
		output2 = ((output[:,1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity).unsqueeze(1)
		#output2 = 0.3 * torch.tanh(out[:, 1]).unsqueeze(1)  # Bound to (-3, 3)

		#rospy.loginfo("Actor output after scaling: %s", str(torch.cat([output1, output2], dim=1)))
		return torch.cat([output1, output2], dim=1)



class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		
		#self.ln_input1 = nn.LayerNorm(state_dim + action_dim)
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 128)
		#self.bn1 = nn.LayerNorm(128)  # Add normalization
		self.l2 = nn.Linear(128 , 64)
		#self.bn2 = nn.LayerNorm(64)  # Add normalization
		self.l3 = nn.Linear(64 , 1)

		#self.ln_input2 = nn.LayerNorm(state_dim + action_dim)
		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 128)
		#self.bn4 = nn.LayerNorm(128)  # Add normalization
		self.l5 = nn.Linear(128 , 64)
		#self.bn5 = nn.LayerNorm(64)  # Add normalization
		self.l6 = nn.Linear(64 , 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		#sa1 = self.ln_input1(sa)
		q1 = F.relu(self.l1(sa))
		#q1 = self.bn1(q1)  # Add normalization after activation
		q1 = F.relu(self.l2(q1))
		#q1 = self.bn2(q1)  # Add normalization after activation
		q1 = self.l3(q1)

		#sa2 = self.ln_input2(sa)
		q2 = F.relu(self.l4(sa))
		#q2 = self.bn4(q2)  # Add normalization after activation
		q2 = F.relu(self.l5(q2))
		#q2 = self.bn5(q2)  # Add normalization after activation
		q2 = self.l6(q2)
		
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		#sa = self.ln_input1(sa)
		q1 = F.relu(self.l1(sa))
		#q1 = self.bn1(q1)  # Add normalization
		q1 = F.relu(self.l2(q1))
		#q1 = self.bn2(q1)  # Add normalization
		q1 = self.l3(q1)
		
		return q1
	
class PathNetwork(nn.Module):
		def __init__(self, input_dim):
			super(PathNetwork, self).__init__()
			
			#self.ln_input1 = nn.LayerNorm(state_dim + action_dim)
			# Q1 architecture
			self.l1 = nn.Linear(input_dim , 128)
			#self.bn1 = nn.LayerNorm(128)  # Add normalization
			self.l2 = nn.Linear(128 , 64)
			#self.bn2 = nn.LayerNorm(64)  # Add normalization
			self.l3 = nn.Linear(64 , 32)

		def forward(self, input):
			#sa = torch.cat([state, action], 1)
			#sa1 = self.ln_input1(sa)
			q1 = F.relu(self.l1(input))
			#q1 = self.bn1(q1)  # Add normalization after activation
			q1 = F.relu(self.l2(q1))
			#q1 = self.bn2(q1)  # Add normalization after activation
			q1 = self.l3(q1)

			return q1


class TD3(nn.Module):
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

		self.ddpg_input_dim = rospy.get_param("/Spatial_Attention/spatial_att_outdim")
		self.spatial_input_size = rospy.get_param("/Spatial_Attention/n_sector_spatialatt")
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.spatial_attention = Spatial_Attention(2*self.spatial_input_size + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)  
		self.temporal_attention = Temporal_Attention(4 + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)
		
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

		if self.att_flag:
			self.actor = Actor(self.ddpg_input_dim , self.action_dim).to(self.device)
			self.critic = Critic(self.ddpg_input_dim , self.action_dim).to(self.device)
		else:
			#self.actor = Actor(632 , self.action_dim).to(self.device)
			#self.critic = Critic(632 , self.action_dim).to(self.device)
			self.actor = Actor(90 + 10 , self.action_dim).to(self.device)
			self.critic = Critic(90 + 10 , self.action_dim).to(self.device)

		# Initialize network weights
		init_network_weights(self.actor)
		init_network_weights(self.critic)

        # Create target networks
		self.actor_target = copy.deepcopy(self.actor)
		self.critic_target = copy.deepcopy(self.critic)

        # Setup optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(rospy.get_param("/Network_param/actor_lr")))		
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(rospy.get_param("/Network_param/critic_lr")), weight_decay=1e-3)

		#self.replay_buffer = Replay_Buffer(rospy.get_param("/Training/buffer_size"))
		if self.att_flag:
			self.replay_buffer = ReplayBuffer(50000 , rospy.get_param("/Training/buffer_size") , 180 , 2 , 2*rospy.get_param("/Training/n_waypoint"))
		else:
			self.replay_buffer = ReplayBuffer(50000 , rospy.get_param("/Training/buffer_size") , 90 , 2 , 2*rospy.get_param("/Training/n_waypoint"))
        # Initialize noise processes for exploration
		self.exploration_noise = OUNoise(self.action_dim)
		self.gaussian_noise = GaussianNoise(self.action_dim)
        
        # Hyperparameters
		self.batch_size = rospy.get_param("/Training/batch_size")
		self.gamma = rospy.get_param("/TD3/discount_factor")
		self.tau = rospy.get_param("/TD3/tau")
		self.policy_noise = rospy.get_param("/TD3/policy_noise")
		self.noise_clip = rospy.get_param("/TD3/noise_clip")
		self.policy_freq = rospy.get_param("/TD3/policy_freq")

		self.total_it = 0
	
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
			# Apply normalization to stabilize the TD3 input
			#normalized_spatial_out = self.spatial_output_norm(spatial_out)
			
			# Get raw action from actor
			action = self.actor(spatial_out).cpu()
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
			raw_action = self.actor(torch.cat((spatial_input, waypoints), dim=1)).cpu()
		#if not add_noise:
			#rospy.loginfo("Raw action: " + str(raw_action))
        # Scale action to actual velocity range
		#if not add_noise:
			#rospy.loginfo("Action: " + str(action))
		
        # Add exploration noise if required
		if add_noise:
			noise = torch.tensor(self.gaussian_noise.noise()).view(1, 2)
			action = action + noise    
		else: rospy.loginfo(str(action))
		action = action.view(-1)
   
        # Clamp action to allowed range
		action[0] = torch.clamp(action[0], min=self.min_linear_velocity, max=self.max_linear_velocity)
		action[1] = torch.clamp(action[1], min=-self.max_angular_velocity, max=self.max_angular_velocity)

		return action 

	def update(self, path):

		if int(self.replay_buffer.count()) < 500:
			return
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward , next_state , waypoints, next_waypoints, is_done = self.replay_buffer.sample()

		if self.att_flag:

			# Get spatial attention outputs
			curr_spatial_out = self.spatial_attention(state, waypoints)
			next_spatial_out = self.spatial_attention(next_state, next_waypoints)
			
			# Apply normalization to stabilize TD3 inputs
			#curr_spatial_out = self.spatial_output_norm(curr_spatial_out)
			#next_spatial_out = self.spatial_output_norm(next_spatial_out)
		else:

			curr_spatial_out = torch.cat((state, waypoints), dim=1)
			next_spatial_out = torch.cat((next_state, next_waypoints), dim=1)
			#curr_spatial_out = torch.cat((state, extra_input), dim=1)
			#next_spatial_out = torch.cat((next_state, next_extra_input), dim=1)
		
		with torch.no_grad():

            # Add noise for target policy smoothing
			next_action = self.actor_target(next_spatial_out)
			#noise = torch.tensor(self.exploration_noise.noise()).view(1, 2).to(self.device)
			# Select action according to policy and add clipped noise
			noise = torch.clamp(torch.randn_like(action) * 0.2, -0.5, 0.5)
			#noise = noise.clamp(-self.noise_clip, self.noise_clip)
			next_action = next_action + noise
			
			# Normalize and scale actions
			#normalized_next_action = torch.zeros_like(next_action)
			#normalized_next_action[:, 0] = (next_action[:, 0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity
			#normalized_next_action[:, 1] = (next_action[:, 1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity

			#normalized_next_action += noise
			
			# Clamp actions
			next_action[:, 0] = torch.clamp(next_action[:, 0], 
                                               min=self.min_linear_velocity, 
                                               max=self.max_linear_velocity)
			next_action[:, 1] = torch.clamp(next_action[:, 1], 
                                               min=-self.max_angular_velocity, 
                                               max=self.max_angular_velocity)
			
		# Compute target Q values
		target_Q1, target_Q2 = self.critic_target(next_spatial_out, next_action)
		target_Q = torch.min(target_Q1, target_Q2)
		rospy.loginfo("Target Q: " + str(target_Q))
		target_Q = reward + is_done * self.gamma * target_Q
		with open(path + "/critic_q.txt", 'a') as file:
			file.write(str(target_Q.mean()) + "\n")

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(curr_spatial_out, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		with open(path + "/critic_loss.txt", 'a') as file:
			file.write(str(critic_loss.item()) + "\n")

		with open(path + "/critic_diff.txt", 'a') as file:
			value_diff = torch.abs(current_Q1 - current_Q2)
			mean_diff = torch.mean(value_diff).item()
			file.write(str(mean_diff) + "\n")

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward(retain_graph=True)
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
		torch.nn.utils.clip_grad_norm_(self.spatial_attention.parameters(), max_norm=10.0)
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			actor_loss = -self.critic.Q1(curr_spatial_out, self.actor(curr_spatial_out)).mean()

			with open(path + "/actor_loss.txt", 'a') as file:
				file.write(str(actor_loss.item()) + "\n")

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
			torch.nn.utils.clip_grad_norm_(self.spatial_attention.parameters(), max_norm=10.0)
			self.actor_optimizer.step()

			rospy.loginfo(self.check_gradients(self.spatial_attention))
			rospy.loginfo(self.check_gradients(self.actor))
			rospy.loginfo(self.check_gradients(self.critic))

			# Update target networks
			self.soft_upgrade_target()


	def soft_upgrade_target(self):
		# Soft update of target networks
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	#def print_model(self):


	# Diagnostic function
	def check_gradients(self, model):
		rospy.logwarn(str(model))
		for name, param in model.named_parameters():
			if param.grad is not None:
				rospy.logwarn(f"{name}: grad norm = {param.grad.norm().item()}")
			else:
				rospy.logwarn(f"{name}: No gradient")

	def save(self, filename, folder_path):
		#Save all model componenets
		try:
			os.makedirs(folder_path, exist_ok=True)
			torch.save(self.critic.state_dict(), os.path.join(folder_path, filename + "_critic"))
			torch.save(self.critic_optimizer.state_dict(), os.path.join(folder_path, filename + "_critic_optimizer"))

			torch.save(self.actor.state_dict(), os.path.join(folder_path, filename + "_actor"))
			torch.save(self.actor_optimizer.state_dict(), os.path.join(folder_path, filename + "_actor_optimizer"))

			torch.save(self.spatial_attention.state_dict(), os.path.join(folder_path, filename + "_spatial_attention"))
			#torch.save(self.spatial_attention_optimizer.state_dict(), os.path.join(folder_path, filename + "_spatialatt_optimizer"))
        
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
			self.critic_target = copy.deepcopy(self.critic)
            
            # Load actor
			self.actor.load_state_dict(torch.load(os.path.join(folder_path, filename + "_actor")))
			self.actor_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_actor_optimizer")))
			self.actor_target = copy.deepcopy(self.actor)
            
            # Load spatial attention
			self.spatial_attention.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatial_attention")))
			#self.spatial_attention_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatialatt_optimizer")))
            
            # Load normalization layer if it exists
			#try:
				#self.spatial_output_norm.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatial_output_norm")))
			#except FileNotFoundError:
				#rospy.logwarn("No saved spatial_output_norm found, using initialized values")
                
			rospy.loginfo(f"Model successfully loaded from {folder_path}")
		except Exception as e:
			rospy.logerr(f"Error loading model: {str(e)}")
	
	def update_buffer(self , state , new_state , reward , action , waypoints , next_waypoints , extra_input , next_extra_input , is_done):
		#state = self.laser_scan_norm(state)
		#new_state = self.laser_scan_norm(new_state)
		self.replay_buffer.record((state, action, reward, new_state , waypoints , next_waypoints , is_done))


class TD3_2(nn.Module):
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

		self.ddpg_input_dim = rospy.get_param("/Spatial_Attention/spatial_att_ourdim")
		self.spatial_input_size = rospy.get_param("/Spatial_Attention/n_sector_spatialatt")
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


		#network definition 
				
		self.spatial_attention = Spatial_Attention(2*self.spatial_input_size + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)  
		self.temporal_attention = Temporal_Attention(4 + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)

		self.path_net = PathNetwork(2*rospy.get_param("/Training/n_waypoint")).to(self.device)

		self.actor = Actor(self.ddpg_input_dim + 32, self.action_dim).to(self.device)
		self.critic = Critic(self.ddpg_input_dim + 32, self.action_dim).to(self.device)

		score_init_network_weights(self.path_net)
		# Initialize network weights
		init_network_weights(self.actor)
		init_network_weights(self.critic)

        # Create target networks
		self.actor_target = copy.deepcopy(self.actor)
		self.critic_target = copy.deepcopy(self.critic)

        # Setup optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(rospy.get_param("/Network_param/actor_lr")))		
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(rospy.get_param("/Network_param/critic_lr")), weight_decay=1e-3)

		#self.replay_buffer = Replay_Buffer(rospy.get_param("/Training/buffer_size"))
		self.replay_buffer = ReplayBuffer(50000 , rospy.get_param("/Training/buffer_size") , 180 , 2 , 2*rospy.get_param("/Training/n_waypoint"))
		
        # Initialize noise processes for exploration
		self.exploration_noise = OUNoise(self.action_dim)
		self.gaussian_noise = GaussianNoise(self.action_dim)
        
        # Hyperparameters
		self.batch_size = rospy.get_param("/Training/batch_size")
		self.gamma = rospy.get_param("/TD3/discount_factor")
		self.tau = rospy.get_param("/TD3/tau")
		self.policy_noise = rospy.get_param("/TD3/policy_noise")
		self.noise_clip = rospy.get_param("/TD3/noise_clip")
		self.policy_freq = rospy.get_param("/TD3/policy_freq")

		self.total_it = 0
	
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
			# Apply normalization to stabilize the TD3 input
			#normalized_spatial_out = self.spatial_output_norm(spatial_out)
			
			# Get raw action from actor
			raw_action = self.actor(spatial_out).cpu()
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
			raw_action = self.actor(torch.cat((spatial_input, waypoints), dim=1)).cpu()
		if not add_noise:
			rospy.loginfo("Raw action: " + str(raw_action))
        # Scale action to actual velocity range
		action = torch.zeros(2)
		action[0] = (raw_action[0][0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity
		action[1] = (raw_action[0][1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity
		if not add_noise:
			rospy.loginfo("Action: " + str(action))
		
        # Add exploration noise if required
		if add_noise:
			noise = torch.tensor(self.gaussian_noise.noise()).view(1, 2)
			action = action + noise    
		action = action.view(-1)
   
        # Clamp action to allowed range
		action[0] = torch.clamp(action[0], min=self.min_linear_velocity, max=self.max_linear_velocity)
		action[1] = torch.clamp(action[1], min=-self.max_angular_velocity, max=self.max_angular_velocity)

		return action 

	def update(self, path):

		if int(self.replay_buffer.count()) < 500:
			return
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward , next_state , waypoints, next_waypoints, is_done = self.replay_buffer.sample()

		if self.att_flag:

			# Get spatial attention outputs
			curr_spatial_out = self.spatial_attention(state, waypoints)
			next_spatial_out = self.spatial_attention(next_state, next_waypoints)
			
			# Apply normalization to stabilize TD3 inputs
			#curr_spatial_out = self.spatial_output_norm(curr_spatial_out)
			#next_spatial_out = self.spatial_output_norm(next_spatial_out)
		else:

			curr_spatial_out = torch.cat((state, waypoints), dim=1)
			next_spatial_out = torch.cat((next_state, next_waypoints), dim=1)
			#curr_spatial_out = torch.cat((state, extra_input), dim=1)
			#next_spatial_out = torch.cat((next_state, next_extra_input), dim=1)
		
		with torch.no_grad():

            # Add noise for target policy smoothing
			next_action = self.actor_target(next_spatial_out)
			#noise = torch.tensor(self.exploration_noise.noise()).view(1, 2).to(self.device)
			# Select action according to policy and add clipped noise
			noise = torch.clamp(torch.randn_like(action) * 0.2, -0.5, 0.5)
			#noise = noise.clamp(-self.noise_clip, self.noise_clip)
			next_action = next_action + noise
			
			# Normalize and scale actions
			#normalized_next_action = torch.zeros_like(next_action)
			#normalized_next_action[:, 0] = (next_action[:, 0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity
			#normalized_next_action[:, 1] = (next_action[:, 1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity

			#normalized_next_action += noise
			
			# Clamp actions
			next_action[:, 0] = torch.clamp(next_action[:, 0], 
                                               min=self.min_linear_velocity, 
                                               max=self.max_linear_velocity)
			next_action[:, 1] = torch.clamp(next_action[:, 1], 
                                               min=-self.max_angular_velocity, 
                                               max=self.max_angular_velocity)
			
		# Compute target Q values
		target_Q1, target_Q2 = self.critic_target(next_spatial_out, next_action)
		target_Q = torch.min(target_Q1, target_Q2)
		target_Q = reward + is_done * self.gamma * target_Q
		with open(path + "/critic_q.txt", 'a') as file:
			file.write(str(target_Q.mean()) + "\n")

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(curr_spatial_out, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		with open(path + "/critic_loss.txt", 'a') as file:
			file.write(str(critic_loss.item()) + "\n")

		with open(path + "/critic_diff.txt", 'a') as file:
			value_diff = torch.abs(current_Q1 - current_Q2)
			mean_diff = torch.mean(value_diff).item()
			file.write(str(mean_diff) + "\n")

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward(retain_graph=True)
		#torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			actor_loss = -self.critic.Q1(curr_spatial_out, self.actor(curr_spatial_out)).mean()

			with open(path + "/actor_loss.txt", 'a') as file:
				file.write(str(actor_loss.item()) + "\n")

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			#torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
			self.actor_optimizer.step()

			rospy.loginfo(self.check_gradients(self.spatial_attention))
			rospy.loginfo(self.check_gradients(self.actor))
			rospy.loginfo(self.check_gradients(self.critic))

			# Update target networks
			self.soft_upgrade_target()


	def soft_upgrade_target(self):
		# Soft update of target networks
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	#def print_model(self):


	# Diagnostic function
	def check_gradients(self, model):
		rospy.logwarn(str(model))
		for name, param in model.named_parameters():
			if param.grad is not None:
				rospy.logwarn(f"{name}: grad norm = {param.grad.norm().item()}")
			else:
				rospy.logwarn(f"{name}: No gradient")

	def save(self, filename, folder_path):
		#Save all model componenets
		try:
			os.makedirs(folder_path, exist_ok=True)
			torch.save(self.critic.state_dict(), os.path.join(folder_path, filename + "_critic"))
			torch.save(self.critic_optimizer.state_dict(), os.path.join(folder_path, filename + "_critic_optimizer"))

			torch.save(self.actor.state_dict(), os.path.join(folder_path, filename + "_actor"))
			torch.save(self.actor_optimizer.state_dict(), os.path.join(folder_path, filename + "_actor_optimizer"))

			torch.save(self.spatial_attention.state_dict(), os.path.join(folder_path, filename + "_spatial_attention"))
			#torch.save(self.spatial_attention_optimizer.state_dict(), os.path.join(folder_path, filename + "_spatialatt_optimizer"))
        
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
			self.critic_target = copy.deepcopy(self.critic)
            
            # Load actor
			self.actor.load_state_dict(torch.load(os.path.join(folder_path, filename + "_actor")))
			self.actor_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_actor_optimizer")))
			self.actor_target = copy.deepcopy(self.actor)
            
            # Load spatial attention
			self.spatial_attention.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatial_attention")))
			#self.spatial_attention_optimizer.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatialatt_optimizer")))
            
            # Load normalization layer if it exists
			#try:
				#self.spatial_output_norm.load_state_dict(torch.load(os.path.join(folder_path, filename + "_spatial_output_norm")))
			#except FileNotFoundError:
				#rospy.logwarn("No saved spatial_output_norm found, using initialized values")
                
			rospy.loginfo(f"Model successfully loaded from {folder_path}")
		except Exception as e:
			rospy.logerr(f"Error loading model: {str(e)}")
	
	def update_buffer(self , state , new_state , reward , action , waypoints , next_waypoints , extra_input , next_extra_input , is_done):
		#state = self.laser_scan_norm(state)
		#new_state = self.laser_scan_norm(new_state)
		self.replay_buffer.record((state, action, reward, new_state , waypoints , next_waypoints , is_done))