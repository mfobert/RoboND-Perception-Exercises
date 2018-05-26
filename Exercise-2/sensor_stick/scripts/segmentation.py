#!/usr/bin/env python

# Import modules
from pcl_helper import *

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    #Convert ROS msg to PCL data
    pcl_input_cloud = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    voxel_downsampler = pcl_input_cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01 #voxel size
    voxel_downsampler.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_voxel_downsampled = voxel_downsampler.filter()

    # PassThrough Filter
    #z
    passthrough = cloud_voxel_downsampled.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = .6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered_z = passthrough.filter()
    #y
    passthrough_y = cloud_filtered_z.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_y.set_filter_field_name(filter_axis)
    axis_min = -20
    axis_max = -1.4
    passthrough_y.set_filter_limits(axis_min, axis_max)
    cloud_filtered_yz = passthrough_y.filter()

    # RANSAC Plane Segmentation
    segmenter = cloud_filtered_yz.make_segmenter()
    segmenter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.005
    segmenter.set_distance_threshold(max_distance)
    inliers, coefficients = segmenter.segment()

    # Extract inliers/outliers
    ros_cloud_table = cloud_filtered_yz.extract(inliers, negative=False)
    ros_cloud_objects = cloud_filtered_yz.extract(inliers, negative=True)
    
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(ros_cloud_objects)

    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(10000)

    tree = white_cloud.make_kdtree()
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()


    #Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    
    # Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(ros_cloud_table)
    ros_cloud_objects = pcl_to_ros(ros_cloud_objects)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)
    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)
    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    # Initialize color_list
    get_color_list.color_list = []

    #Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
