from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import math
import numpy as np
import rospy

class TAGD:
    def __init__(self ):
        self.n_iterations = rospy.get_param("/ICP/n_iter")
        self.threshold = rospy.get_param("/ICP/threshold")
        self.relative_rmse = rospy.get_param("/ICP/error_thresh")
        self.source_for_icp = None
        self.Nc = rospy.get_param("/TAGD/Nc")
        self.d_thresh = rospy.get_param("/TAGD/d_thresh")

    def TAGD(self , prev_scan , curr_scan):
        angle_step = 0.33 #agle step of laser scan use in tiago robot for define the scan that we need to analyze
        theta_thresh = round(90/(self.Nc * 0.33)) #range of angle to analyze around the theta_ref
        icp_prev_scan = self.ICP(prev_scan , curr_scan)
        #self.laser_plot(icp_prev_scan, curr_scan)
        if len(icp_prev_scan) != len(prev_scan):
           rospy.loginfo("Erorr in ICP algo")
        # Convert curr_scan to distances (scalar float values)
        distances = np.linalg.norm(curr_scan, axis=1)
        tagd_list = []
        for i in range(self.Nc):
          theta_ref = round((180*(i+1))/(self.Nc*angle_step))
          #for idx in range(theta_ref - theta_thresh , theta_ref + theta_thresh):
            # Slice the distances array from (x - n) to (x + n)
          start_idx = max(0, theta_ref - theta_thresh)  # Ensure the index is valid
          end_idx = min(len(distances), theta_ref + theta_thresh)  # Ensure the range does not exceed array bounds

          subset = distances[start_idx:end_idx]  # Extract the range
          min_idx_in_subset = np.argmin(subset)  # Index of the minimum in the subset
          min_idx = start_idx + min_idx_in_subset  # Convert back to the original array index
          #for theta in rage()
          min_ray = curr_scan[min_idx]
          # Calculate distances between min_ray and all points in prev_scan and curr_scan
          distances_prev_scan = np.linalg.norm(icp_prev_scan - min_ray, axis=1)
          distances_curr_scan = np.linalg.norm(curr_scan - min_ray, axis=1)
          # Filter points within the threshold distance
          filtered_prev_scan = icp_prev_scan[distances_prev_scan < self.d_thresh]
          filtered_curr_scan = curr_scan[distances_curr_scan < self.d_thresh]
          # Calculate centroids of the filtered points
          centroid_prev_scan = (
            np.mean(filtered_prev_scan, axis=0) if filtered_prev_scan.size > 0 else None
          )
          centroid_curr_scan = (
            np.mean(filtered_curr_scan, axis=0) if filtered_curr_scan.size > 0 else None
          )
          if centroid_prev_scan is None or centroid_curr_scan is None:
            rospy.logerr("error in centroid")

          # Store only the centroids (not the filtered points)
          #tagd_list.append({
          #  'centroid_prev_scan': centroid_prev_scan,
          #  'centroid_curr_scan': centroid_curr_scan,
          #})
          tagd_list.append(np.concatenate((centroid_prev_scan, centroid_curr_scan), axis=0))

        return tagd_list , icp_prev_scan

    def ICP(self , source_points , target_points):

        # Ensure input is numpy array
        source_points = np.asarray(source_points)
        target_points = np.asarray(target_points)
        # Convert Nx2 to Nx3 by adding a zero z-coordinate
        source_points_3d = np.hstack((source_points, np.zeros((source_points.shape[0], 1))))
        target_points_3d = np.hstack((target_points, np.zeros((target_points.shape[0], 1))))
        source_indices = []
        target_indices = []
        prev_rmse = 0
        self.source_for_icp = source_points_3d

        # Transformation matrix (initialize as identity)
        transformation = np.eye(4)

        # ICP main loop
        for iter in range(0 , self.n_iterations):

            #find closest point
            source_indices , target_indices , rmse = self.find_closest_point(target_points_3d)

            #use svd implementation of ICP
            transformation = np.dot(transformation , self.get_svd_icp_transformation(target_points_3d , source_indices , target_indices))

            rospy.logdebug("At iteration number : " + str(iter+1) + " , the error value is : " + str(rmse) + " with a difference respect the previous value of : " + str(prev_rmse - rmse))
            if abs(prev_rmse - rmse) < self.relative_rmse :
              break
            else:
              prev_rmse = rmse


            #apply transformation to the source pointfor the next iteration
            #source_for_icp_ = source_;
            self.source_for_icp = source_points_3d
            self.source_for_icp = self.apply_transformation(source_points_3d , transformation)


        # Remove the third dimension (z = 0) to return to 2D
        transformed_source_points_2d = self.source_for_icp[:, :2]
        return transformed_source_points_2d


    def apply_transformation(self , points, transformation):

        # Convert to homogeneous coordinates
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))

        # Apply the transformation
        transformed_points_h = np.dot(transformation, points_h.T).T

        # Convert back to Cartesian coordinates
        return transformed_points_h[:, :3]

    def find_closest_point(self , target_points ):

      #source_points = np.asarray(source_points)
      target_points = np.asarray(target_points)
      # Build the k-d tree
      kdtree = cKDTree(target_points)

      # Initialize lists to store indices and RMSE
      target_indices = []
      source_indices = []
      rmse = 0.0
      # Iterate through source points
      for count, source_point in enumerate(self.source_for_icp):
          # Search for nearest neighbors
          # query returns (distances, indices)
          distances, indices = kdtree.query(source_point)

          # Ensure we're working with scalar distance for single nearest neighbor
          distances = np.atleast_1d(distances)
          indices = np.atleast_1d(indices)

          # Control if the distance found is less than the threshold
          if distances[0] <= self.threshold:
              # Save the indices and calculate the error
              target_indices.append(indices[0])
              source_indices.append(count)

              # Update RMSE incrementally
              rmse = rmse * count / (count + 1) + distances[0] / (count + 1)
      if len(target_indices) == 0 :
        rospy.logerr("error in target indeces dimension")
      if len(target_indices) == 0 :
        rospy.logerr("error in target indeces dimension")
      return target_indices, source_indices, rmse
    def get_svd_icp_transformation(self , target_points , source_indices , target_indices):

        # Initialize the transformation matrix as identity
        transformation = np.eye(4)

        # Calculate the centroids of the source and target point clouds
        centroid_source = np.mean(self.source_for_icp[source_indices], axis=0)
        centroid_target = np.mean(target_points[target_indices], axis=0)
        # Center the points by subtracting the centroids
        centered_source = self.source_for_icp[source_indices] - centroid_source
        centered_target = target_points[target_indices] - centroid_target

        # Compute the matrix W for SVD
        W = np.dot(centered_target.T, centered_source)

        # Perform Singular Value Decomposition (SVD) on W
        U, _, Vt = np.linalg.svd(W)

        # Compute the rotation matrix R
        R = np.dot(U, Vt)
        # Manage the special reflection case
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1  # Flip the sign of the third column

        # Compute the translation vector t
        t = centroid_target - np.dot(R.copy(), centroid_source)
        # Populate the transformation matrix
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        #control if thr transformation matrix has the correct dimension
        if transformation.shape != (4, 4):
           rospy.logerr("Transformation matrix dimension error : " + str(transformation.shape))

        return transformation
    
    def laser_plot(self , source , target):
      # Plotting
      plt.figure(figsize=(8, 6))
      plt.scatter(source[:, 0], source[:, 1], label='Source', s=5)
      plt.scatter(target[:, 0], target[:, 1], label='Target', s=5)
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.title('Cartesian Point Cloud')
      plt.legend()
      plt.grid(True)
      plt.show()