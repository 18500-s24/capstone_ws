#include <Eigen/Dense> /* Eigen */

#include <rclcpp/rclcpp.hpp>                /* obvious */
#include <sensor_msgs/msg/point_cloud2.hpp> /* sensor_msgs::msg::PointCloud2 */
#include <std_msgs/msg/u_int8_multi_array.hpp> /* std_msgs::msg::UInt8MultiArray */
#include <std_srvs/srv/trigger.hpp>            /* std_srvs::srv::Trigger */

#include <octomap/octomap.h>          /* octomap */
#include <octomap_msgs/conversions.h> /* octomap_msgs::fullMapToMsg */

#include <pcl/common/transforms.h>           /* pcl::transformPointCloud */
#include <pcl/point_cloud.h>                 /* pcl::PointCloud */
#include <pcl/point_types.h>                 /* pcl::PointXYZ */
#include <pcl_conversions/pcl_conversions.h> /* pcl_conversions::fromROSMsg */

#include <cmath>   /* sin, cos, sqrt, pow */
#include <fstream> /* std::ifstream */
#include <math.h>  /* M_PI */

class KinectOctomapNode : public rclcpp::Node {
  public:
    KinectOctomapNode() : Node("kinect_octomap_node") {
        // subscribers
        point_cloud_subscriber_ =
            this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/kinect2/hd/points",
                rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(
                                rmw_qos_profile_sensor_data))
                    .best_effort(),
                std::bind(&KinectOctomapNode::pointCloudCallback, this,
                          std::placeholders::_1));

        // publishers
        intarray_publisher_ =
            this->create_publisher<std_msgs::msg::UInt8MultiArray>(
                "/kinect2_map/intarray", 10);

        // parameters
        this->declare_parameter<bool>("calibrated", false);
        this->declare_parameter<double>("octree_resolution", 0.01);
        this->declare_parameter<std::string>("output_path", "~/tree.bt");
        this->declare_parameter<double>("x_rotation", 240.0);
        this->declare_parameter<double>("y_rotation", 0.0);
        this->declare_parameter<double>("z_rotation", 0.0);
        this->declare_parameter<double>("x_translation", 0.0);
        this->declare_parameter<double>("y_translation", 0.0);
        this->declare_parameter<double>("z_translation", 0.0);
        this->declare_parameter<double>("scene_x_max", 1.28);  // in meters
        this->declare_parameter<double>("scene_y_max", 0.64);  // in meters
        this->declare_parameter<double>("scene_z_max", 0.64);  // in meters
        this->declare_parameter<double>("arm_base_x", 0.61);   // in meters
        this->declare_parameter<double>("arm_base_y", 0.0);    // in meters
        this->declare_parameter<double>("arm_base_z", 0.0);    // in meters
        this->declare_parameter<double>("arm_max_readh", 0.5); // in meters

        // service to save the octree
        save_octree_service_ = this->create_service<std_srvs::srv::Trigger>(
            "save_octree",
            std::bind(&KinectOctomapNode::saveOctreeCallback, this,
                      std::placeholders::_1, std::placeholders::_2));
    }

    ~KinectOctomapNode() {
        if (current_tree != nullptr) {
            delete current_tree;
            current_tree = nullptr;
        }
    }

  private:
    // subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        point_cloud_subscriber_;

    // publishers
    rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr
        intarray_publisher_;

    // services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_octree_service_;

    octomap::OcTree *current_tree = nullptr;

    /**
     * @brief Compose a transformation matrix based on the parameters
     */
    Eigen::Matrix4f prepareTransformationMatrix() {
        // get rotation and translation parameters
        float x_rot_deg =
            static_cast<float>(this->get_parameter("x_rotation").as_double());
        float y_rot_deg =
            static_cast<float>(this->get_parameter("y_rotation").as_double());
        float z_rot_deg =
            static_cast<float>(this->get_parameter("z_rotation").as_double());
        float x_trans = static_cast<float>(
            this->get_parameter("x_translation").as_double());
        float y_trans = static_cast<float>(
            this->get_parameter("y_translation").as_double());
        float z_trans = static_cast<float>(
            this->get_parameter("z_translation").as_double());

        // convert degrees to radians for rotation
        float x_rot_rad = x_rot_deg * M_PI / 180.0f;
        float y_rot_rad = y_rot_deg * M_PI / 180.0f;
        float z_rot_rad = z_rot_deg * M_PI / 180.0f;

        // create rotation matrices
        Eigen::Matrix4f rx = Eigen::Matrix4f::Identity();
        rx(1, 1) = cos(x_rot_rad);
        rx(1, 2) = -sin(x_rot_rad);
        rx(2, 1) = sin(x_rot_rad);
        rx(2, 2) = cos(x_rot_rad);
        Eigen::Matrix4f ry = Eigen::Matrix4f::Identity();
        ry(0, 0) = cos(y_rot_rad);
        ry(0, 2) = sin(y_rot_rad);
        ry(2, 0) = -sin(y_rot_rad);
        ry(2, 2) = cos(y_rot_rad);
        Eigen::Matrix4f rz = Eigen::Matrix4f::Identity();
        rz(0, 0) = cos(z_rot_rad);
        rz(0, 1) = -sin(z_rot_rad);
        rz(1, 0) = sin(z_rot_rad);
        rz(1, 1) = cos(z_rot_rad);

        // combine rotations
        Eigen::Matrix4f rotation_matrix = rz * ry * rx;

        // create translation matrix
        Eigen::Matrix4f translation_matrix = Eigen::Matrix4f::Identity();
        translation_matrix(0, 3) = x_trans;
        translation_matrix(1, 3) = y_trans;
        translation_matrix(2, 3) = z_trans;

        // Ccombine rotation and translation into a single transformation matrix
        Eigen::Matrix4f transform = translation_matrix * rotation_matrix;

        return transform;
    }

    /**
     * @brief Crop the point cloud to the scene dimensions
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr
    cropPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);

        float scene_x_min = 0.0;
        float scene_y_min = 0.0;
        float scene_z_min = 0.0;
        float scene_x_max =
            static_cast<float>(this->get_parameter("scene_x_max").as_double());
        float scene_y_max =
            static_cast<float>(this->get_parameter("scene_y_max").as_double());
        float scene_z_max =
            static_cast<float>(this->get_parameter("scene_z_max").as_double());

        for (const auto &point : cloud->points) {
            if (point.x >= scene_x_min && point.x <= scene_x_max &&
                point.y >= scene_y_min && point.y <= scene_y_max &&
                point.z >= scene_z_min && point.z <= scene_z_max) {
                cropped_cloud->push_back(point);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Cropped point cloud to scene limits.");

        return cropped_cloud;
    }

    /**
     * @brief Prune the unreachable regions from the octree
     */
    void pruneUnreachableRegions(octomap::OcTree *tree) {
        float scene_x_max =
            static_cast<float>(this->get_parameter("scene_x_max").as_double());
        float scene_y_max =
            static_cast<float>(this->get_parameter("scene_y_max").as_double());
        float scene_z_max =
            static_cast<float>(this->get_parameter("scene_z_max").as_double());
        float arm_base_x =
            static_cast<float>(this->get_parameter("arm_base_x").as_double());
        float arm_base_y =
            static_cast<float>(this->get_parameter("arm_base_y").as_double());
        float arm_base_z =
            static_cast<float>(this->get_parameter("arm_base_z").as_double());
        float arm_max_reach = static_cast<float>(
            this->get_parameter("arm_max_readh").as_double());
        for (float x = 0.005; x < scene_x_max; x += 0.005) {
            for (float y = 0.005; y < scene_y_max; y += 0.005) {
                for (float z = 0.005; z < scene_z_max; z += 0.005) {
                    float dist =
                        sqrt(pow(x - arm_base_x, 2) + pow(y - arm_base_y, 2) +
                             pow(z - arm_base_z, 2));
                    if (dist > arm_max_reach) {
                        tree->updateNode(octomap::point3d(x, y, z), true);
                    }
                }
            }
        }
    }

    /**
     * @brief Fill the back of the objects in the octree
     */
    void fillBackOfObjects(octomap::OcTree *tree) {
        float scene_x_max =
            static_cast<float>(this->get_parameter("scene_x_max").as_double());
        float scene_y_max =
            static_cast<float>(this->get_parameter("scene_y_max").as_double());
        float scene_z_max =
            static_cast<float>(this->get_parameter("scene_z_max").as_double());

        // for each voxel in the x-z plane with y >= y_threshold, if the voxel
        // is occupied, fill all the voxels behind it
        float y_threshold = 0.15;
        for (float x = 0.005; x < scene_x_max; x += 0.005) {
            for (float z = 0.005; z < scene_z_max; z += 0.005) {
                bool found_occupied = false;
                float y_occupied = 0.0;
                for (float y = y_threshold; y < scene_y_max; y += 0.005) {
                    octomap::OcTreeNode *node = tree->search(x, y, z);
                    if (node != nullptr) {
                        if (node->getOccupancy() > 0.5) {
                            found_occupied = true;
                            y_occupied = y;
                            break;
                        }
                    }
                }
                if (found_occupied) {
                    for (float y = y_occupied; y < scene_y_max; y += 0.005) {
                        octomap::OcTreeNode *node = tree->search(x, y, z);
                        if (node != nullptr) {
                            if (node->getOccupancy() > 0.5) {
                                break;
                            } else {
                                tree->updateNode(octomap::point3d(x, y, z),
                                                 true);
                            }
                        }
                    }
                }
            }
        }

        RCLCPP_INFO(this->get_logger(), "Filled back of objects in octree.");
    }

    /**
     * @brief Publish the octree as UInt8MultiArray
     */
    void publishUInt8MultiArray(
        octomap::OcTree *tree,
        rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr
            publisher) {
        std_msgs::msg::UInt8MultiArray intarray_msg;
        float scene_x_max =
            static_cast<float>(this->get_parameter("scene_x_max").as_double());
        float scene_y_max =
            static_cast<float>(this->get_parameter("scene_y_max").as_double());
        float scene_z_max =
            static_cast<float>(this->get_parameter("scene_z_max").as_double());
        for (float z = 0.005; z < scene_z_max; z += 0.005) {
            for (float y = 0.005; y < scene_y_max; y += 0.005) {
                for (float x = 0.005; x < scene_x_max; x += 0.005) {
                    octomap::OcTreeNode *node = tree->search(x, y, z);
                    if (node != nullptr) {
                        if (node->getOccupancy() > 0.5) {
                            intarray_msg.data.push_back(1);
                        } else {
                            intarray_msg.data.push_back(0);
                        }
                    } else {
                        RCLCPP_ERROR(this->get_logger(),
                                     "(%f, %f, %f) not found in octree.", x, y,
                                     z);
                    }
                }
            }
        }
        publisher->publish(intarray_msg);
        RCLCPP_INFO(
            this->get_logger(),
            "Published octree as UInt8MultiArray to /kinect2_map/intarray.");
    }

    void
    pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(),
                    "Received a newly published sensor message.");

        if (this->get_parameter("calibrated").as_bool()) {
            RCLCPP_INFO(this->get_logger(),
                        "Calibration done. Doing pruning and cropping.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Calibration not done.");
        }

        // Reinitialize the octree with a fresh one each time
        if (current_tree != nullptr) {
            delete current_tree;
        }
        current_tree = new octomap::OcTree(
            this->get_parameter("octree_resolution").as_double());

        // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        RCLCPP_INFO(this->get_logger(),
                    "Converted sensor message to pcl point cloud.");

        // Prepare transformation matricx
        Eigen::Matrix4f transformation_matrix = prepareTransformationMatrix();

        // Apply transformation matrix on the pcl point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud, *transformed_cloud,
                                 transformation_matrix);

        // Crop the point cloud if calibration is done
        if (this->get_parameter("calibrated").as_bool()) {
            transformed_cloud = cropPointCloud(transformed_cloud);
        }

        // Convert pcl::PointCloud to octomap
        for (const auto &point : transformed_cloud->points) {
            current_tree->updateNode(
                octomap::point3d(point.x, point.y, point.z), true);
        }
        RCLCPP_INFO(this->get_logger(),
                    "Converted pcl point cloud to octomap.");

        // To avoid the RRT from creating a path through the unreachable
        // regions, we need to set the unreachable regions as occupied
        // Only do this if calibration is done
        if (this->get_parameter("calibrated").as_bool()) {
            pruneUnreachableRegions(current_tree);
        }

        // Mark the back of the objects as occupied (since the kinect cannot
        // see the back of the objects)
        if (this->get_parameter("calibrated").as_bool()) {
            fillBackOfObjects(current_tree);
        }

        // Convert octomap to intarray and publish
        if (this->get_parameter("calibrated").as_bool()) {
            publishUInt8MultiArray(current_tree, intarray_publisher_);
        }
    }

    void
    saveOctreeCallback(const std_srvs::srv::Trigger::Request::SharedPtr request,
                       std_srvs::srv::Trigger::Response::SharedPtr response) {
        std::ifstream tree_file;
        tree_file.open(this->get_parameter("output_path").as_string());

        current_tree->writeBinary(
            this->get_parameter("output_path").as_string());
        response->success = true;
        response->message = "Octree saved successfully.";
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KinectOctomapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
