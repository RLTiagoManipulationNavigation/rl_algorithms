import torch 

import numpy as np
from RL_Algorithms.model import (ActorNet , CriticNet)
from RL_Algorithms.replay_buffer import ReplayBuffer , Replay_Buffer
from RL_Algorithms.attention_module import Spatial_Attention
import rospy
from RL_Algorithms.OUNoise import OUNoise
from RL_Algorithms.utils import clone_model , target_weight_update
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import copy

class DDPG(nn.Module):
    def __init__(self, env , actor_param , critic_param , spatial_flag):
        super().__init__()
        self.spatial_flag = spatial_flag
        self.actor_param = actor_param
        self.critic_param = critic_param
        self.name = 'DDPG' # name for uploading results
        self.environment = env
        self.ddpg_input_dim = rospy.get_param("/Spatial_Attention/spatial_att_ourdim")
        self.spatial_input_size = rospy.get_param("/Spatial_Attention/input_spatial_size")

        #linear and angular velocity raange
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')

        #if rospy.get_param("/DDPG/seed") > 0:
        #self.seed('12345')

        #Setting environment param
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        #initialize Spatial Attention Network
        self.spatial_attention = Spatial_Attention(self.spatial_input_size + 10)

        #initialize critic and actor network , giving in input dimension of input and output layer
        if self.spatial_flag :
            self.actor_network = ActorNet(self.ddpg_input_dim)
            self.critic_network = CriticNet(self.ddpg_input_dim + self.action_dim)
        else:
            self.actor_network = ActorNet(self.state_dim)
            self.critic_network = CriticNet(self.state_dim + self.action_dim)

        #initialize target networks 
        #self.target_actor = clone_model(self.actor_network)
        #self.target_critic = clone_model(self.critic_network)
        self.target_actor = copy.deepcopy(self.actor_network)
        self.target_critic = copy.deepcopy(self.critic_network)

        
        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        # initialize replay buffer
        #self.replay_buffer = ReplayBuffer(self.state_dim , self.action_dim)
        self.replay_buffer = Replay_Buffer(rospy.get_param("/Training/buffer_size"))
        self.replay_buffer.erase()
        #initialize the optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=actor_param['lr'], weight_decay=actor_param['weight_decay'])
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters() ,  lr=critic_param['lr'], weight_decay=critic_param['weight_decay'])

        self.target_update = rospy.get_param("/DDPG/soft_target_update")

        self.discount_factor = rospy.get_param("/DDPG/discount_factor")

        self.batch_size = actor_param['batch_size']

        self.flag = True

    def forward(self , input , waypoints , add_noise = True):

        if self.spatial_flag :
            attention_process = self.spatial_attention(input , waypoints)
            #rospy.loginfo("Lidar : " + str(input))
            #rospy.loginfo("ATTENTION OUTPUT : " + str(attention_process))
            action = self.actor_network(attention_process)
        else:    
            if isinstance(input, list):
                input = torch.tensor(input, dtype=torch.float32)
            if len(input.shape) == 1:
                input = input.unsqueeze(0)  # Shape becomes (1, input_dim)    
            action = self.actor_network(input)
        
        if add_noise:
            #generate OUNoise 
            noise = torch.tensor(self.exploration_noise.noise()).view(1, 2)
            #rospy.loginfo("noise : " + str(noise))
            action = action + noise
            self.flag = True
        else:
            if self.flag : 
                rospy.loginfo("Actor :")
                for param in self.actor_network.parameters():
                    rospy.loginfo("parameter")
                    rospy.loginfo(str(param))
                rospy.loginfo("Critic :")
                for param in self.critic_network.parameters():
                    rospy.loginfo("parameter")
                    rospy.loginfo(str(param))
    
                self.flag = False    
        action = action.view(-1)  # Reshape if necessary    
        # Assuming `action` is a tensor with two elements
        action[0] = torch.clamp(action[0], min=self.min_linear_velocity, max=self.max_linear_velocity)  # Clip first value between 0.1 and 0.6
        action[1] = torch.clamp(action[1], min=-self.max_angular_velocity, max=self.max_angular_velocity)  # Clip second value between -0.5 and 0.8
        return action 
    
    def update(self , pkg_path):
        if int(self.replay_buffer.count()) < self.batch_size :
            return
        # Sample replay buffer 
        #state, action, next_state, reward, waypoints = self.replay_buffer.sample(self.batch_size)
        state, action, next_state, reward, waypoints = self.replay_buffer.sample(self.batch_size)

        spatial_state = self.spatial_attention(state , waypoints)
        #rospy.loginfo("State batch :" + str(state_batch))
        spatial_next_state = self.spatial_attention(next_state , waypoints)

        if self.spatial_flag:
            target_Q = self.target_critic(spatial_next_state, self.target_actor(spatial_next_state))
        else:
		    # Compute the target Q value
            target_Q = self.target_critic(next_state, self.target_actor(next_state))
        #rospy.loginfo(str(target_Q))
        target_Q = reward + (self.discount_factor * target_Q).detach()

        if self.spatial_flag:
		    # Get current Q estimate
            current_Q = self.critic_network(spatial_state, action)
        else:    
            current_Q = self.critic_network(state, action)
        #self.save_data(pkg_path , "/critic/output_value"+ str(self.actor_param) +"_"+ str(self.critic_param)+".txt" , current_Q)
        #self.save_data(pkg_path , "/critic/weight_value"+ str(self.actor_param) +"_"+ str(self.critic_param)+".txt" , self.critic_network.parameters())      
        #self.save_data(pkg_path , "/actor/weight_value"+ str(self.actor_param) +"_"+ str(self.critic_param)+".txt" , self.actor_network.parameters())    
        #self.save_data(pkg_path , "/critic/target_weight_value"+ str(self.actor_param) +"_"+ str(self.critic_param)+".txt" , self.target_critic.parameters())      
        #self.save_data(pkg_path , "/actor/target_weight_value"+ str(self.actor_param) +"_"+ str(self.critic_param)+".txt" , self.target_actor.parameters())      
        if self.spatial_flag:          
            self.save_data(pkg_path , "/spatial/weight_value"+ str(self.actor_param) +"_"+ str(self.critic_param)+".txt" , self.spatial_attention.parameters()) 
         
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        if self.spatial_flag:
		    # Compute actor loss
            actor_loss = -self.critic_network(spatial_state, self.actor_network(spatial_state)).mean()
        else:    
            actor_loss = -self.critic_network(state, self.actor_network(state)).mean()
		
		# Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        #self.check_gradients(self.actor_network)
        # Update the frozen target models
        for param, target_param in zip(self.critic_network.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.target_update * param.data + (1 - self.target_update) * target_param.data)

        for param, target_param in zip(self.actor_network.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.target_update * param.data + (1 - self.target_update) * target_param.data)

        #self.save_data(pkg_path , "/critic/gradient_value"+ str(self.actor_param) +"_"+ str(self.critic_param)+".txt" , self.check_gradients(self.critic_network))      
        #self.save_data(pkg_path , "/actor/gradient_value"+ str(self.actor_param) +"_"+ str(self.critic_param)+".txt" , self.check_gradients(self.actor_network))
         # Update target networks
        #target_weight_update(self.target_actor, self.actor_network, self.target_update)
        #target_weight_update(self.target_critic, self.critic_network, self.target_update)
    
    # Diagnostic function
    def check_gradients(self , model):
        rospy.logwarn(str(model))
        for name, param in model.named_parameters():
            if param.grad is not None:
                rospy.logwarn(f"{name}: grad norm = {param.grad.norm().item()}")
            else:
                rospy.logwarn(f"{name}: No gradient")

    def save_data(self , pkg_path , file_name , data):
        full_path = os.path.join(pkg_path, file_name)
        #os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path , 'a') as file:
            # Append the new data to the end of the file
            rospy.loginfo("print")
            file.write(str(data) + "\n")              

    def update_buffer(self , state , new_state , reward , action , waypoints):
        self.replay_buffer.add(state, action, new_state, reward, waypoints)
        #self.replay_buffer.add( state, action , reward, new_state , waypoints)   

    def save(self, filename , folder_path):
        torch.save(self.critic_network.state_dict(),  os.path.join(folder_path, filename + "_critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(folder_path,filename + "_critic_optimizer"))
        
        torch.save(self.actor_network.state_dict(), os.path.join(folder_path,filename + "_actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder_path,filename + "_actor_optimizer"))

        torch.save(self.spatial_attention.state_dict(), os.path.join(folder_path,filename + "_spatial_attention"))
        #torch.save(self.attention_optimizer.state_dict(), os.path.join(folder_path,filename + "_spatial_attention_optimizer"))
        
    def load(self, filename , folder_path):
        self.critic_network.load_state_dict(torch.load(os.path.join(folder_path,filename + "_critic")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(folder_path,filename + "_critic_optimizer")))
        self.critic_target = clone_model(self.critic_network)
        
        self.actor_network.load_state_dict(torch.load(os.path.join(folder_path,filename + "_actor")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(folder_path,filename + "_actor_optimizer")))
        self.actor_target = clone_model(self.actor_network)

        self.spatial_attention.load_state_dict(torch.load(os.path.join(folder_path,filename + "_spatial_attention")))
    
        #self.attention_optimizer.load_state_dict(torch.load(os.path.join(folder_path,filename + "_spatial_attention_optimizer")))

    def seed(self,s):
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        #if USE_CUDA:
        #    torch.cuda.manual_seed(s)    
     




