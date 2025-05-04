import torch 
import numpy as np
import rospy
import rospkg
import torch.nn as nn
from RL_Algorithms.model import Embedding, Feature, Score
import torch.nn.functional as F
from RL_Algorithms.utils import *
import matplotlib.pyplot as plt

class Spatial_Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = 'Spatial Attention'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_section = rospy.get_param("/Spatial_Attention/input_spatial_size")
        self.output_dim = rospy.get_param("/Spatial_Attention/embedding_output_size")
        
        # Values for generating the correct version of laser scan 
        self.initial_angle = rospy.get_param("/Tiago/initial_angle")
        self.angle_increment = rospy.get_param("/Tiago/angle_increment")
        self.n_discard_scan = rospy.get_param("/Tiago/remove_scan")
        self.nrays = rospy.get_param("/Spatial_Attention/n_rays")
        
        self.Embedding = Embedding(input_dim)
        self.Score = Score(self.output_dim)
        self.Feature = Feature(self.output_dim)
        self.test = rospy.get_param("Training/test")
        
        # Initialize network weights
        score_init_network_weights(self.Embedding)
        init_network_weights(self.Score)
        init_network_weights(self.Feature)
        
        self.debug = rospy.get_param("/Training/debug")
        self.algorithm_name = rospy.get_param("/Training/algorithm")
        # Set the logging system
        self.rospack = rospkg.RosPack()

    def forward(self, input, waypoints , eval = False):
        
        # Ensure input has batch dimension
        if len(input.shape) == 1:
            input = input.unsqueeze(0)  # Shape becomes (1, input_dim)
        
        # Ensure waypoints has batch dimension
        if len(waypoints.shape) == 1:
            waypoints = waypoints.unsqueeze(0)  # Shape becomes (1, input_dim) 
        # Ensure batch sizes match - broadcast if needed
        if waypoints.shape[0] != input.shape[0]:
            waypoints = waypoints.expand(input.shape[0], -1)
            
        input_dim = input.shape[1] // 2  # input dimension (e.g., lidar readings per sample)
        
        # Compute section size (each sample is divided into n_section parts)
        section_size = int(2 * (input_dim // self.n_section))
        
        # Split input for each sample into sections
        sections = torch.chunk(input, self.n_section, dim=1)
        
        embeddings = []
        scores = []
        features = []
        
        for i in range(self.n_section):
            section = sections[i]
            # Process each section in the batch
            ei = self.Embedding(torch.cat((section, waypoints), dim=1))  # Apply the embedding layer
            embeddings.append(ei)
            
            si = self.Score(ei)  # Apply the score layer
            scores.append(si)
            
            fi = self.Feature(ei)  # Apply the feature layer
            features.append(fi)
        
        # Stack along a new dimension (representing the sections)
        embeddings = torch.stack(embeddings, dim=1)  # Shape: (batch_size, n_section, output_dim)
        scores = torch.stack(scores, dim=1)  # Shape: (batch_size, n_section, 1)
        features = torch.stack(features, dim=1)  # Shape: (batch_size, n_section, output_dim)
        
        # Softmax normalization of scores across sections
        attention_weights = F.softmax(scores, dim=1)  # Normalize across sections, Shape: (batch_size, n_section, 1)
        
        #rospy.loginfo(str(attention_weights))
        if self.debug and eval:
            with open(self.rospack.get_path('tiago_navigation') + "/data/" + str(self.algorithm_name) + "_attention_score.txt", 'a') as file:  
                for i in range(attention_weights.shape[1]):  # Batch dimension
                    file.write(str(attention_weights[0, i, 0].item()))
                    if i < attention_weights.shape[1] - 1:
                        file.write(",")
                file.write("\n")
        
        # Weighted sum of features across sections
        weighted_features = features * attention_weights  # Element-wise multiplication, broadcasting over output_dim
        
        # Sum across the sections (dim=1), leaving (batch_size, output_dim)
        output = torch.sum(weighted_features, dim=1)
        
        #output = torch.tanh(output)

        if rospy.get_param("Training/debug", False):
            rospy.logdebug("output shape: " + str(output.shape))
            
        return output
    

    ##########################################################################
    ## SPATIAL ATTENTION FOR SUBGOAL-DRIVEN NAVIGATION #######################
    ##########################################################################
    
    def spatial_wout_waypoints(self, input, eval = False):
        
        # Ensure input has batch dimension
        if len(input.shape) == 1:
            input = input.unsqueeze(0)  # Shape becomes (1, input_dim)
        
        # Split input for each sample into sections
        sections = torch.chunk(input, self.n_section, dim=1)
        
        embeddings = []
        scores = []
        features = []
        
        for i in range(self.n_section):
            section = sections[i]
            #rospy.loginfo("section " + str(torch.cat((section, waypoints), dim=1)))
            # Process each section in the batch
            ei = self.Embedding(section)  # Apply the embedding layer
            embeddings.append(ei)
            
            si = self.Score(ei)  # Apply the score layer
            scores.append(si)
            
            fi = self.Feature(ei)  # Apply the feature layer
            features.append(fi)
        
        # Stack along a new dimension (representing the sections)
        embeddings = torch.stack(embeddings, dim=1)  # Shape: (batch_size, n_section, output_dim)
        scores = torch.stack(scores, dim=1)  # Shape: (batch_size, n_section, 1)
        features = torch.stack(features, dim=1)  # Shape: (batch_size, n_section, output_dim)
        
        # Softmax normalization of scores across sections
        attention_weights = F.softmax(scores, dim=1)  # Normalize across sections, Shape: (batch_size, n_section, 1)
        
        #rospy.loginfo(str(attention_weights))
        if self.debug and eval:
            with open(self.rospack.get_path('tiago_navigation') + "/data/" + str(self.algorithm_name) + "_attention_score.txt", 'a') as file:  
                for i in range(attention_weights.shape[1]):  # Batch dimension
                    file.write(str(attention_weights[0, i, 0].item()))
                    if i < attention_weights.shape[1] - 1:
                        file.write(",")
                file.write("\n")
        
        # Weighted sum of features across sections
        weighted_features = features * attention_weights  # Element-wise multiplication, broadcasting over output_dim
        
        # Sum across the sections (dim=1), leaving (batch_size, output_dim)
        output = torch.sum(weighted_features, dim=1)
        
        #output = torch.tanh(output)

        if rospy.get_param("Training/debug", False):
            rospy.logdebug("output shape: " + str(output.shape))
            
        return output
    

#######################################
#                                     #
#       ATTENTION ARCHITECTURE        #
#                                     #
#######################################    

#Embedding Network
class Attention_Module(nn.Module):
  def __init__(self, input_size):
    super().__init__()

    #EMBEDDING LAYER     
    #input layer
    self.embedding_l1 = nn.Linear(input_size, 256)
    #self.bn1 = nn.LayerNorm(256)  
    
    #hidden layer
    self.embedding_l2 = nn.Linear(256, 128)
    #self.bn2 = nn.LayerNorm(128)  
    
    #output layer
    self.embedding_l3 = nn.Linear(128, 64)

    #FEATURE LAYER
    #input layer
    self.feature_l1 = nn.Linear(64, 80)
    #self.bn1 = nn.LayerNorm(80) 
    
    #hidden layer
    self.feature_l2 = nn.Linear(80, 50)
    #self.bn2 = nn.LayerNorm(50)  
    
    #output layer
    self.feature_l3 = nn.Linear(50, 30)

    #SCORE NETWORK
    #input layer
    self.score_l1 = nn.Linear(64, 60)
    #self.ln1 = nn.LayerNorm(80) 
    
    #hidden layer
    self.score_l2 = nn.Linear(60, 50)
    #self.ln2 = nn.LayerNorm(50)  
    
    #output layer
    self.score_l3 = nn.Linear(50, 1)

    self.init_weights()
        
  def init_weights(self):
    # Fan-in initialization for linear layers
    nn.init.xavier_uniform_(self.embedding_l1.weight)
    nn.init.xavier_uniform_(self.embedding_l2.weight)
    nn.init.xavier_uniform_(self.embedding_l3.weight)
    nn.init.xavier_uniform_(self.feature_l1.weight)
    nn.init.xavier_uniform_(self.feature_l2.weight)
    nn.init.xavier_uniform_(self.feature_l3.weight)
    nn.init.xavier_uniform_(self.score_l1.weight)
    nn.init.xavier_uniform_(self.score_l2.weight)
    nn.init.xavier_uniform_(self.score_l3.weight)

    nn.init.constant_(self.embedding_l1.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.embedding_l2.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.embedding_l3.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.feature_l1.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.feature_l2.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.feature_l3.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.score_l1.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.score_l2.bias, 0.01)  # Small positive bias
    nn.init.constant_(self.score_l3.bias, 0.01)  # Small positive bias
    
  def forward(self, input):
    
    # EMBEDDING NETWORK
    a = F.relu(self.embedding_l1(input))
    
    a = F.relu(self.embedding_l2(a))

    a = F.relu(self.embedding_l3(a))

    # FEATURE NETWORK
    feature = F.relu(self.feature_l1(a))
    feature = F.relu(self.feature_l2(feature))
    #feature = F.relu(self.feature_l3(feature))
    feature = self.feature_l3(feature)

    # SCORE NETWORK
    score = F.relu(self.score_l1(a))
    score = F.relu(self.score_l2(score))
    #score = F.relu(self.score_l3(score))
    score = self.score_l3(score)

    
    return feature , score 
  

class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = 'Spatial Attention'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_section = rospy.get_param("/Spatial_Attention/input_spatial_size")
        self.output_dim = rospy.get_param("/Spatial_Attention/embedding_output_size")
        
        # Values for generating the correct version of laser scan 
        self.initial_angle = rospy.get_param("/Tiago/initial_angle")
        self.angle_increment = rospy.get_param("/Tiago/angle_increment")
        self.n_discard_scan = rospy.get_param("/Tiago/remove_scan")
        self.nrays = rospy.get_param("/Spatial_Attention/n_rays")
        
        self.Attention = Attention_Module(input_dim)
        self.Attention = initialize_network(self.Attention)

        self.test = rospy.get_param("Training/test")
        
        # Initialize network weights
        #score_init_network_weights(self.Attention)
        
        self.debug = rospy.get_param("/Training/debug")
        self.algorithm_name = rospy.get_param("/Training/algorithm")
        # Set the logging system
        self.rospack = rospkg.RosPack()

    def forward(self, input, waypoints , eval = False):
        
        # Ensure input has batch dimension
        if len(input.shape) == 1:
            input = input.unsqueeze(0)  # Shape becomes (1, input_dim)
        
        # Ensure waypoints has batch dimension
        if len(waypoints.shape) == 1:
            waypoints = waypoints.unsqueeze(0)  # Shape becomes (1, input_dim) 
        # Ensure batch sizes match - broadcast if needed
        if waypoints.shape[0] != input.shape[0]:
            waypoints = waypoints.expand(input.shape[0], -1)
            
        input_dim = input.shape[1] // 2  # input dimension (e.g., lidar readings per sample)
        
        # Compute section size (each sample is divided into n_section parts)
        section_size = int(2 * (input_dim // self.n_section))
        
        # Split input for each sample into sections
        sections = torch.chunk(input, self.n_section, dim=1)
        
        scores = []
        features = []
        
        for i in range(self.n_section):
            section = sections[i]
            # Process each section in the batch
            fi , si = self.Attention(torch.cat((section, waypoints), dim=1))  # Apply the embedding layer
            
            scores.append(si)
            features.append(fi)
        
        # Stack along a new dimension (representing the sections)
        scores = torch.stack(scores, dim=1)  # Shape: (batch_size, n_section, 1)
        features = torch.stack(features, dim=1)  # Shape: (batch_size, n_section, output_dim)
        
        # Softmax normalization of scores across sections
        attention_weights = F.softmax(scores, dim=1)  # Normalize across sections, Shape: (batch_size, n_section, 1)

        
        #rospy.loginfo(str(attention_weights))
        if self.debug and eval:
            with open(self.rospack.get_path('tiago_navigation') + "/data/" + str(self.algorithm_name) + "_attention_score.txt", 'a') as file:  
                for i in range(attention_weights.shape[1]):  # Batch dimension
                    file.write(str(attention_weights[0, i, 0].item()))
                    if i < attention_weights.shape[1] - 1:
                        file.write(",")
                file.write("\n")
        
        # Weighted sum of features across sections
        weighted_features = features * attention_weights  # Element-wise multiplication, broadcasting over output_dim
    
        # Sum across the sections (dim=1), leaving (batch_size, output_dim)
        output = torch.sum(weighted_features, dim=1)

        output = torch.tanh(output)  # Bounded output
        
        #output = torch.tanh(output)

        if rospy.get_param("Training/debug", False):
            rospy.logdebug("output shape: " + str(output.shape))
            
        return output



  
