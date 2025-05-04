import torch 

import numpy as np
from RL_Algorithms.model import ActorNet , CriticNet , SpatioTemporalActor , SpatioTemporalCritic
from RL_Algorithms.replay_buffer import ReplayBuffer , Replay_Buffer
from RL_Algorithms.attention_module import Spatial_Attention , Attention
import rospy
from RL_Algorithms.OUNoise import OUNoise , GaussianNoise , OUActionNoise
from RL_Algorithms.utils import *
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import copy
import random

class DDPG(nn.Module):
    def __init__(self, env):
        super().__init__()
		#Setting environment param
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.num_sectors = rospy.get_param('/Spatial_Attention/input_spatial_size')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		        
        self.actor = SpatioTemporalActor( self.action_dim ).to(self.device)
        self.critic = SpatioTemporalCritic( self.action_dim ).to(self.device)

        self.actor_target = SpatioTemporalActor( self.action_dim ).to(self.device)
        self.critic_target = SpatioTemporalCritic( self.action_dim ).to(self.device)
        
        #self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 0.0001 , eps = 1e-6) #lr=float(rospy.get_param('/Network_param/actor_lr')) , eps=rospy.get_param('/Network_param/eps'))

        #self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 0.0001 , eps = 1e-6) #lr=float(rospy.get_param('/Network_param/critic_lr')) , eps=rospy.get_param('/Network_param/eps'))

        # Learning rate schedulers
        #self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1, gamma=0.1)
        #self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1, gamma=0.1)

        self.replay_buffer = ReplayBuffer(50000 , rospy.get_param("/Training/batch_size") 
                                          , 2*rospy.get_param("/Spatial_Attention/n_rays") 
                                          , 2 
                                          , 2*rospy.get_param("/Training/n_waypoint"))

        # Create OU noise generator with separate parameters for each dimension
        self.exploration_noise = OUNoise(self.action_dim)
        #self.exploration_noise = OUActionNoise()


        #self.exploration_noise = OUNoise(self.action_dim)
        self.gaussian_noise = GaussianNoise(self.action_dim)
        self.batch_size = rospy.get_param("/Training/batch_size")
        self.tau = 0.02
        self.gamma = rospy.get_param("/DDPG/discount_factor")

        #linear and angular velocity raange
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')

        self.soft_update_count = 0
        self.soft_update_interval = 10  # Update every 10 steps
        self.rand_noise_it = 0

        self.check_weight(self.actor)
        self.check_weight(self.actor_target)
        self.check_weight(self.critic)
        self.check_weight(self.critic_target)

    def forward(self , spatial_input , waypoints , goal_pos , add_noise = True):
        
		# Ensure input is a tensor (convert if necessary)
		#input = self.laser_scan_norm(input)
        # Ensure input is a tensor (convert if necessary)
        spatial_input = torch.from_numpy(np.array(spatial_input)).float()
        spatial_input = torch.tensor(spatial_input, dtype=torch.float32).to(self.device)
		#temporal_input = torch.tensor(temporal_input, dtype=torch.float32).to(self.device)
        waypoints = torch.FloatTensor(np.array(waypoints)).to(self.device)
		#waypoints = torch.tensor(waypoints , dtype=torch.float32).to(self.device)
        goal_pos = torch.FloatTensor(np.array(goal_pos)).to(self.device).unsqueeze(0)

        #spatial_input = self.input_split(spatial_input , waypoints)
        if add_noise:
            action = self.actor(spatial_input , waypoints).cpu()
        else:
            action = self.actor(spatial_input , waypoints , True).cpu()

        if add_noise :
            noise = torch.tensor(self.gaussian_noise.noise()).view(1, 2)
            #rospy.loginfo("noise : " + str(noise))
            action = action + noise
        #else: rospy.loginfo(str(action))
        action = action.view(-1)
		#action = action.view(-1)  # Reshape if necessary    
        # Assuming `action` is a tensor with two elements
        action[0] = torch.clamp(action[0], min=-1, max=1)  # Clip first value between 0.1 and 0.6
        action[1] = torch.clamp(action[1], min=-1, max=1)  # Clip second value between -0.5 and 0.8
        
        #action[0] = torch.clamp(action[0], min=self.min_linear_velocity, max=self.max_linear_velocity)  # Clip first value between 0.1 and 0.6
        #action[1] = torch.clamp(action[1], min=-self.max_angular_velocity, max=self.max_angular_velocity)  # Clip second value between -0.5 and 0.8
        """if self.rand_noise_it <= 2000:
            if add_noise != True:
                print("inside")
            action[0] = random.uniform(self.min_linear_velocity, self.max_linear_velocity)
            action[1] = random.uniform(self.min_angular_velocity, self.max_angular_velocity)
            self.rand_noise_it += 1"""
        return action
    
    def update(self, path):
        #torch.autograd.set_detect_anomaly(True)  # Debugging tool

        if self.replay_buffer.count() < 100:
            return
       
        state, action, reward , next_state , waypoints, next_waypoints, is_done = self.replay_buffer.sample()

        target_action = self.actor_target(next_state , next_waypoints)
        target_critic = self.critic_target(next_state , next_waypoints , target_action)
        critic_value = self.critic(state , waypoints , action)

        target = []
        for i in range(self.batch_size):
            target.append(reward[i] + (is_done[i] * self.gamma * target_critic[i]))
        target = torch.tensor(target).to(self.device)
        target = target.view(self.batch_size , 1)

        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(target , critic_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Loss ---
        self.actor_optimizer.zero_grad()
        actions_pred = self.actor(state , waypoints)
        actor_loss = -self.critic(state , waypoints , actions_pred)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_param()

    def update_network_param(self):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic_params = self.critic_target.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = self.tau * critic_state_dict[name].clone() + (1 - self.tau) * target_critic_dict[name].clone()

        self.critic_target.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = self.tau * actor_state_dict[name].clone() + (1 - self.tau) * target_actor_dict[name].clone()
        self.actor_target.load_state_dict(actor_state_dict)
    def check_weight(self , model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                print(f"{name} | Mean: {param.data.mean().item():.4f}, Std: {param.data.std().item():.4f}")  
        
    def update_target(self , target, original):
        for target_param, param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	# Diagnostic function
    def check_gradients(self , model):
        rospy.logwarn(str(model))
        for name, param in model.named_parameters():
            if param.grad is not None:
                rospy.logwarn(f"{name}: grad norm = {param.grad.norm().item()}")
            else:
                rospy.logwarn(f"{name}: No gradient")
    
    def check_gradients_update(self , params_before , model):
        # Compare parameters after the update
        updated = False
        for p_before, p_after in zip(params_before, model.parameters()):
            if not torch.equal(p_before, p_after.data):
                updated = True
                break
        if updated:
            rospy.loginfo("Model parameters have been updated.")
        else:
            rospy.loginfo("Model parameters have not been updated.")

    
    def save(self, filename , folder_path):
        torch.save(self.critic.state_dict(),  os.path.join(folder_path, filename + "_critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(folder_path,filename + "_critic_optimizer"))
        
        torch.save(self.actor.state_dict(), os.path.join(folder_path,filename + "_actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder_path,filename + "_actor_optimizer"))
        
        #torch.save(self.spatial_attention.state_dict(), os.path.join(folder_path,filename + "_spatial_attention"))

    def load(self, filename , folder_path):
        self.critic.load_state_dict(torch.load(os.path.join(folder_path,filename + "_critic")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(folder_path,filename + "_critic_optimizer")))
        self.critic_target = clone_model(self.critic)
        
        self.actor.load_state_dict(torch.load(os.path.join(folder_path,filename + "_actor")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(folder_path,filename + "_actor_optimizer")))
        self.actor_target = clone_model(self.actor)		

        #self.spatial_attention.load_state_dict(torch.load(os.path.join(folder_path,filename + "_spatial_attention")))
	
    def update_buffer(self , state , new_state , reward , action , waypoints , next_waypoints , extra_input , next_extra_input , is_done):
		#state = self.laser_scan_norm(state)
		#new_state = self.laser_scan_norm(new_state)
        self.replay_buffer.record((state, action, reward, new_state , waypoints , next_waypoints , is_done))

    def print_param(self):
        rospy.loginfo("Actor :")
        for param in self.actor_network.parameters():
            rospy.loginfo("parameter")
            rospy.loginfo(str(param))
        rospy.loginfo("Critic :")
        for param in self.critic_network.parameters():
            rospy.loginfo("parameter")
            rospy.loginfo(str(param))

    def input_split(self , lidar_data , waypoints):
        # Ensure input has batch dimension
        if len(lidar_data.shape) == 1:
            lidar_data = lidar_data.unsqueeze(0)  # Shape becomes (1, input_dim)
        
        # Ensure waypoints has batch dimension
        if len(waypoints.shape) == 1:
            waypoints = waypoints.unsqueeze(0)  # Shape becomes (1, input_dim) 
        
        lidar_data = lidar_data.reshape(lidar_data.shape[0], self.num_sectors, -1)
        waypoints = waypoints.view(lidar_data.shape[0], 1, -1).repeat(1, self.num_sectors, 1)

        # Concatenate sector data and waypoint input
        spatial_input = torch.cat([lidar_data, waypoints], dim=2)
        
        return spatial_input





"""class DDPG(nn.Module):
    def __init__(self, env):
        super().__init__()
		#Setting environment param
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

		#Set the structure of the model used 
        self.spatial_flag = rospy.get_param("/Architecture_modifier/spatial_att")

        self.ddpg_input_dim = rospy.get_param("/Spatial_Attention/spatial_att_outdim")
        #self.spatial_input_size = rospy.get_param("/Spatial_Attention/input_spatial_size")
        self.spatial_input_size = rospy.get_param("/Spatial_Attention/n_sector_spatialatt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
        self.spatial_attention = Attention(2*self.spatial_input_size + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)
        
		#self.temporal_attention = Temporal_Attention(4 + 2*rospy.get_param("/Training/n_waypoint")).to(self.device)
		

		#linear and angular velocity raange
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')


		#value for generate the correct version of laser scan 
        self.initial_angle = rospy.get_param("/Tiago/initial_angle")
        self.angle_increment = rospy.get_param("/Tiago/angle_increment")
        self.n_discard_scan = rospy.get_param("/Tiago/remove_scan")
        
        self.actor = ActorNet( self.ddpg_input_dim , self.action_dim ).to(self.device)
        self.critic = CriticNet( self.ddpg_input_dim , self.action_dim ).to(self.device)
        #self.actor = initialize_network(self.actor)
        #self.critic = initialize_network(self.critic)
        #init_network_weights(self.actor)
        #init_network_weights(self.critic)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(rospy.get_param('/Network_param/actor_lr')))

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(rospy.get_param('/Network_param/critic_lr')))
        self.spatial_attention_target = copy.deepcopy(self.spatial_attention)  
        self.spatial_attention_optimizer = torch.optim.Adam(
            self.spatial_attention.parameters(), 
            lr=1e-4  # Match actor/critic learning rates
        )
        rospy.loginfo(str(env.observation_space.shape[0]))
        self.replay_buffer = ReplayBuffer(50000 , 128 , 240 , 2 , 2*rospy.get_param("/Training/n_waypoint"))


       
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(rospy.get_param('/Network_param/actor_lr')))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(rospy.get_param("/Network_param/critic_lr")))
        #self.spatial_attention_optimizer = torch.optim.Adam(self.spatial_attention.parameters(), lr=float(rospy.get_param("/Network_param/attention_lr")))
		#self.spatial_attention_optimizer = torch.optim.Adam(self.spatial_attention.parameters(), lr=3e-4)
        #for param in self.spatial_attention.parameters():
        #    rospy.loginfo(str(param.requires_grad))
		#self.replay_buffer = ReplayBuffer(self.state_dim , self.action_dim )

		# Initialize a random process the Ornstein-Uhlenbeck process for action exploration

        # OU parameters configuration
        linear_params = {
            'mu': 0.0,
            'theta': 0.15,   # Mean reversion strength (linear)
            'sigma': 0.1     # Noise volatility (linear)
        }

        angular_params = {
            'mu': 0.0,
            'theta': 0.15,   # Mean reversion strength (angular)
            'sigma': 0.5     # Noise volatility (angular)
        }

        # Create OU noise generator with separate parameters for each dimension
        self.exploration_noise = OUNoise(
            mu=[linear_params['mu'], angular_params['mu']],
            theta=[linear_params['theta'], angular_params['theta']],
            sigma=[linear_params['sigma'], angular_params['sigma']],
            dt=0.01
        )

        #self.exploration_noise = OUNoise(self.action_dim)
        self.gaussian_noise = GaussianNoise(self.action_dim)
        self.batch_size = rospy.get_param("/Training/batch_size")
        self.tau = rospy.get_param("/TD3/tau")
        self.gamma = rospy.get_param("/DDPG/discount_factor")


    def forward(self , spatial_input , waypoints , goal_pos , add_noise = True):
		# Ensure input is a tensor (convert if necessary)
		#input = self.laser_scan_norm(input)
        # Ensure input is a tensor (convert if necessary)
        spatial_input = torch.from_numpy(np.array(spatial_input)).float()
        spatial_input = torch.tensor(spatial_input, dtype=torch.float32).to(self.device)
		#temporal_input = torch.tensor(temporal_input, dtype=torch.float32).to(self.device)
        waypoints = torch.FloatTensor(np.array(waypoints)).to(self.device)
		#waypoints = torch.tensor(waypoints , dtype=torch.float32).to(self.device)
        goal_pos = torch.FloatTensor(np.array(goal_pos)).to(self.device).unsqueeze(0)
		#state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if add_noise:
			# Get spatial attention output
            spatial_out = self.spatial_attention(spatial_input, waypoints)
        else:
            spatial_out = self.spatial_attention(spatial_input, waypoints , True)
		#spatial_out = torch.cat((spatial_out , goal_pos) , dim = 1)
		#rospy.loginfo(str(spatial_out.shape))
		#if self.temporal_flag:
			#temporal_out = self.temporal_attention(temporal_input , waypoints)
			#actor_input = torch.cat((spatial_out, temporal_out), dim=0)
			#raw_action = self.actor(actor_input).cpu()
		#else:
        action = self.actor(spatial_out).cpu()
        #rospy.loginfo("raw action : " + str(action))
	
        if add_noise :
            noise = torch.tensor(self.gaussian_noise.noise()).view(1, 2)
            #rospy.loginfo("noise : " + str(noise))
            action = action + noise
        #else: rospy.loginfo(str(action))
        action = action.view(-1)
		#action = action.view(-1)  # Reshape if necessary    
        # Assuming `action` is a tensor with two elements
        action[0] = torch.clamp(action[0], min=self.min_linear_velocity, max=self.max_linear_velocity)  # Clip first value between 0.1 and 0.6
        action[1] = torch.clamp(action[1], min=-self.max_angular_velocity, max=self.max_angular_velocity)  # Clip second value between -0.5 and 0.8

        return action
    
    def update(self, path):
        #torch.autograd.set_detect_anomaly(True)  # Debugging tool

        if self.replay_buffer.count() < 200:
            return
        rospy.loginfo(str(self.replay_buffer.count()))
        # Sample from the replay buffer
        state, action, reward , next_state , waypoints, next_waypoints, is_done = self.replay_buffer.sample()

        # Store parameters before the update
        params_before = [p.clone().detach() for p in self.critic.parameters()]

        # Compute spatial attention output for current and next states
        curr_spatial_out = self.spatial_attention(state, waypoints)
        next_spatial_out = self.spatial_attention_target(next_state, next_waypoints)

        # Compute target Q-value using the target networks
        with torch.no_grad():
            next_actions = self.actor_target(next_spatial_out)
            #rospy.loginfo(str(next_actions))
            target_q = reward + (is_done * self.gamma * self.critic_target(next_spatial_out, next_actions))
            #rospy.loginfo("target q value : " + str(target_q))

        # Compute current Q-value
        q_values = self.critic(curr_spatial_out, action)
        #rospy.loginfo("vurrent q value : " + str(q_values))

        # Compute critic loss
        critic_loss = nn.MSELoss()(q_values, target_q)
        with open(path + "/critic_loss.txt", 'a') as file:
            file.write(str(critic_loss.item()) + "\n")

        # --- Actor Loss ---
        actions_pred = self.actor(curr_spatial_out)
        actor_loss = -self.critic(curr_spatial_out, actions_pred).mean()

        with open(path + "/actor_loss.txt", 'a') as file:
            file.write(str(actor_loss.item()) + "\n")

        # --- Backward Passes ---
        # Zero gradients for all optimizers
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.spatial_attention_optimizer.zero_grad()

        # Compute gradients for critic and spatial attention
        critic_loss.backward(retain_graph=True)
        
        # Compute gradients for actor and spatial attention
        actor_loss.backward()

        # Clip gradients (optional)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.spatial_attention.parameters(), 1.0)

        # Update all parameters AFTER both backward passes
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        self.spatial_attention_optimizer.step()

        
        self.check_gradients_update(params_before, self.critic)

        # Log gradient values for debugging
        #rospy.loginfo(self.check_gradients(self.spatial_attention))
        #rospy.loginfo(self.check_gradients(self.actor))
        #rospy.loginfo(self.check_gradients(self.critic))
        #rospy.loginfo(self.check_gradients(self.actor_target))

        # Update target networks
        self.update_target(self.actor_target, self.actor)
        self.update_target(self.critic_target, self.critic)
        self.update_target(self.spatial_attention_target, self.spatial_attention)
"""