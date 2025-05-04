import random
import numpy as np
from collections import deque
import rospy
import torch



"""class Replay_Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def sample(self, batch_size):
        # Randomly sample batch_size examples
        sample = random.sample(self.buffer, batch_size)
        curr_obs, actions, next_obs ,rewards , waypoints , next_waypoints , extra_input , next_extra_input , is_done = zip(*sample)
        return (	torch.FloatTensor(np.array(curr_obs)).to(self.device),
        			torch.FloatTensor(np.array(actions)).to(self.device),
                    torch.FloatTensor(np.array(next_obs)).to(self.device),
        			torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device),
                    torch.FloatTensor(np.array(waypoints)).to(self.device),
                    torch.FloatTensor(np.array(next_waypoints)).to(self.device),
                    torch.FloatTensor(np.array(extra_input)).to(self.device),
                    torch.FloatTensor(np.array(next_extra_input)).to(self.device),
                    torch.FloatTensor(np.array(is_done)).unsqueeze(1).to(self.device)
		)
    def size(self):
        return self.buffer_size

    def add(self, state, action, new_state , reward , waypoints , next_waypoints , extra_input , next_extra_input , is_done):
        #if self.size() == rospy.get_param("/Training/buffer_size") :
             #rospy.logerr("complete buffer")
        experience = (state, action, new_state , reward , waypoints , next_waypoints , extra_input , next_extra_input , is_done)
        #rospy.loginfo("State batch :" + str(state) + " next : " + str(experience) )
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0"""

class ReplayBuffer:
    def __init__(self, buffer_capacity, batch_size, state_dim, action_dim, waypoint_dim):
        # Use PyTorch tensors instead of NumPy arrays for faster device transfer
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate tensors on the device to reduce memory transfers
        self.state_buffer = torch.zeros((buffer_capacity, state_dim), 
                                        dtype=torch.float32, 
                                        device=self.device)
        self.action_buffer = torch.zeros((buffer_capacity, action_dim), 
                                         dtype=torch.float32, 
                                         device=self.device)
        self.reward_buffer = torch.zeros((buffer_capacity, 1), 
                                         dtype=torch.float32, 
                                         device=self.device)
        self.next_state_buffer = torch.zeros((buffer_capacity, state_dim), 
                                             dtype=torch.float32, 
                                             device=self.device)
        self.waypoint_buffer = torch.zeros((buffer_capacity, waypoint_dim), 
                                           dtype=torch.float32, 
                                           device=self.device)
        self.next_waypoint_buffer = torch.zeros((buffer_capacity, waypoint_dim), 
                                                dtype=torch.float32, 
                                                device=self.device)
        self.is_done_buffer = torch.zeros((buffer_capacity, 1), 
                                          dtype=torch.float32, 
                                          device=self.device)

    def record(self, obs_tuple):
        # Use modulo for wrapping around buffer
        index = self.buffer_counter % self.buffer_capacity

        # Direct tensor assignment to avoid conversion overhead
        self.state_buffer[index] = torch.tensor(obs_tuple[0], 
                                                dtype=torch.float32, 
                                                device=self.device)
        self.action_buffer[index] = torch.tensor(obs_tuple[1], 
                                                 dtype=torch.float32, 
                                                 device=self.device)
        self.reward_buffer[index] = torch.tensor(obs_tuple[2], 
                                                 dtype=torch.float32, 
                                                 device=self.device)
        self.next_state_buffer[index] = torch.tensor(obs_tuple[3], 
                                                     dtype=torch.float32, 
                                                     device=self.device)
        self.waypoint_buffer[index] = torch.tensor(obs_tuple[4], 
                                                   dtype=torch.float32, 
                                                   device=self.device)
        self.next_waypoint_buffer[index] = torch.tensor(obs_tuple[5], 
                                                        dtype=torch.float32, 
                                                        device=self.device)
        self.is_done_buffer[index] = torch.tensor(obs_tuple[6], 
                                                  dtype=torch.float32, 
                                                  device=self.device)
        
        self.buffer_counter += 1

    def sample(self):
        # Use buffer size tracking to avoid unnecessary min() calls
        record_range = min(self.buffer_counter, self.buffer_capacity)
        
        # Use torch.randperm for more efficient sampling
        batch_indices = torch.randperm(record_range, 
                                       dtype=torch.long, 
                                       device=self.device)[:self.batch_size]
        #rospy.loginfo("Batch indices: " + str(batch_indices))
        return (
            self.state_buffer[batch_indices],
            self.action_buffer[batch_indices],
            self.reward_buffer[batch_indices],
            self.next_state_buffer[batch_indices],
            self.waypoint_buffer[batch_indices],
            self.next_waypoint_buffer[batch_indices],
            self.is_done_buffer[batch_indices]
        )

    def count(self):
        return self.buffer_counter        
    

class Replay_Buffer:
    def __init__(self, buffer_capacity, batch_size, state_dim, action_dim, waypoint_dim):
        # Use PyTorch tensors instead of NumPy arrays for faster device transfer
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate tensors on the device to reduce memory transfers
        self.state_buffer = torch.zeros((buffer_capacity, state_dim), 
                                        dtype=torch.float32, 
                                        device=self.device)
        self.action_buffer = torch.zeros((buffer_capacity, action_dim), 
                                         dtype=torch.float32, 
                                         device=self.device)
        self.reward_buffer = torch.zeros((buffer_capacity, 1), 
                                         dtype=torch.float32, 
                                         device=self.device)
        self.next_state_buffer = torch.zeros((buffer_capacity, state_dim), 
                                             dtype=torch.float32, 
                                             device=self.device)
        self.waypoint_buffer = torch.zeros((buffer_capacity, waypoint_dim), 
                                           dtype=torch.float32, 
                                           device=self.device)
        self.next_waypoint_buffer = torch.zeros((buffer_capacity, waypoint_dim), 
                                                dtype=torch.float32, 
                                                device=self.device)
        self.is_done_buffer = torch.zeros((buffer_capacity, 1), 
                                          dtype=torch.float32, 
                                          device=self.device)

    def record(self, obs_tuple):
        # Use modulo for wrapping around buffer
        index = self.buffer_counter % self.buffer_capacity

        # Direct tensor assignment to avoid conversion overhead
        self.state_buffer[index] = torch.tensor(obs_tuple[0], 
                                                dtype=torch.float32, 
                                                device=self.device)
        self.action_buffer[index] = torch.tensor(obs_tuple[1], 
                                                 dtype=torch.float32, 
                                                 device=self.device)
        self.reward_buffer[index] = torch.tensor(obs_tuple[2], 
                                                 dtype=torch.float32, 
                                                 device=self.device)
        self.next_state_buffer[index] = torch.tensor(obs_tuple[3], 
                                                     dtype=torch.float32, 
                                                     device=self.device)
        self.waypoint_buffer[index] = torch.tensor(obs_tuple[4], 
                                                   dtype=torch.float32, 
                                                   device=self.device)
        self.next_waypoint_buffer[index] = torch.tensor(obs_tuple[5], 
                                                        dtype=torch.float32, 
                                                        device=self.device)
        self.is_done_buffer[index] = torch.tensor(obs_tuple[6], 
                                                  dtype=torch.float32, 
                                                  device=self.device)
        
        self.buffer_counter += 1

    def sample(self):
        # Use buffer size tracking to avoid unnecessary min() calls
        record_range = min(self.buffer_counter, self.buffer_capacity)
        
        # Use torch.randperm for more efficient sampling
        batch_indices = torch.randperm(record_range, 
                                       dtype=torch.long, 
                                       device=self.device)[:self.batch_size]
        #rospy.loginfo("Batch indices: " + str(batch_indices))
        return (
            self.state_buffer[batch_indices],
            self.action_buffer[batch_indices],
            self.reward_buffer[batch_indices],
            self.next_state_buffer[batch_indices],
            self.waypoint_buffer[batch_indices],
            self.next_waypoint_buffer[batch_indices],
            self.is_done_buffer[batch_indices]
        )

    def count(self):
        return self.buffer_counter        


        
