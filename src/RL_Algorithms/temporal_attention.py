import rospy
import numpy as np
import torch
import torch.nn as nn
from RL_Algorithms.model import Embedding , Feature , Score
import torch.nn.functional as F

class Temporal_Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = 'Temporal Attention'
        #self.n_section = rospy.get_param("/Spatial_Attention/n_sector_spatialatt")
    
        
        #self.n_section = rospy.get_param("/Spatial_Attention/input_spatial_size")
        self.Nc = rospy.get_param("/TAGD/Nc")
        self.output_dim = rospy.get_param("/Spatial_Attention/embedding_output_size")
        self.Embedding = Embedding(input_dim)
        self.Score = Score(self.output_dim)
        self.Feature = Feature(self.output_dim)

    def forward(self , tagd , waypoints):

        # Check if input is 1D (single sample); if so, add batch dimension
        if len(tagd.shape) == 2:
            tagd = tagd.unsqueeze(0)  # Shape becomes (1, input_dim)
        # Check if input is 1D (single sample); if so, add batch dimension
        if len(waypoints.shape) == 1:
            waypoints = waypoints.unsqueeze(0)  # Shape becomes (1, input_dim)
        rospy.loginfo(str(tagd.shape))
        if tagd.shape[1] != self.Nc or tagd.shape[2] != 1 :
            rospy.logerr("Dimension of Temporal Attention input incorrect!")

        # Expand waypoints to match the number of rows (Nc) in tagd for each batch
        waypoints_expanded = waypoints.expand(tagd.shape[0], self.Nc, -1)  # Shape becomes [batch_size, Nc, 10]
        rospy.loginfo(str(waypoints_expanded.shape))
        # Concatenate along the last dimension
        results = torch.cat((tagd, waypoints_expanded), dim=2)  # Shape becomes [batch_size, Nc, 12]
        rospy.loginfo(str(results.shape))
        # Reshape to [batch_size, 15, 14] if Nc = 15
        results = results.view(tagd.shape[0], self.Nc, -1)
        rospy.loginfo(str(results.shape))
        # Initialize outputs
        embeddings, scores, features = [], [], []

        for result in results:
            rospy.loginfo(str(result))

            ei = self.Embedding(result)  # Apply the embedding layer
            embeddings.append(ei)
            #rospy.loginfo(" embedding : " + str(ei))
            si = self.Score(ei)  # Apply the score layer
            scores.append(si)
            #rospy.loginfo(" score : " + str(si))
            fi = self.Feature(ei)  # Apply the feature layer
            features.append(fi)
        
        # Stack along a new dimension (representing the sections) for each of embeddings, scores, features
        embeddings = torch.stack(embeddings, dim=1)  # Shape: (batch_size, n_section, output_dim)
        scores = torch.stack(scores, dim=1)  # Shape: (batch_size, n_section, 1)
        features = torch.stack(features, dim=1)  # Shape: (batch_size, n_section, output_dim)
        
        # Softmax normalization of scores across sections
        attention_weights = F.softmax(scores, dim=1)  # Normalize across sections, Shape: (batch_size, n_section, 1)
        
        # Weighted sum of features across sections
        weighted_features = features * attention_weights  # Element-wise multiplication, broadcasting over output_dim
        #rospy.logdebug("weighted features: " + str(weighted_features) + " tensor size: " + str(weighted_features.shape))
        #rospy.loginfo(str(weighted_features.shape))
        
        # Sum across the sections (dim=1), leaving (batch_size, output_dim)
        output = torch.sum(weighted_features, dim=1)
        #rospy.logdebug("output: " + str(output) + " tensor size: " + str(output.shape))
        
        return output