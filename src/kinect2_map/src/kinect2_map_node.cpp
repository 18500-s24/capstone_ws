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

#include <cmath>   /* sin, cos */
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
        octomap_publisher_ = this->create_publisher<octomap_msgs::msg::Octomap>(
            "/kinect2_map/octomap", 10);
        intarray_publisher_ =
            this->create_publisher<std_msgs::msg::UInt8MultiArray>(
                "/kinect2_map/intarray", 10);

        // parameters
        this->declare_parameter<double>("octree_resolution", 0.01);
        this->declare_parameter<std::string>("output_path", "~/tree.bt");
        this->declare_parameter<double>("x_rotation", 0.0);
        this->declare_parameter<double>("y_rotation", 0.0);
        this->declare_parameter<double>("z_rotation", 0.0);
        this->declare_parameter<double>("x_translation", 0.0);
        this->declare_parameter<double>("y_translation", 0.0);
        this->declare_parameter<double>("z_translation", 0.0);

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
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_publisher_;
    rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr
        intarray_publisher_;

    // services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_octree_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr calibrate_base_service_;

    octomap::OcTree *current_tree = nullptr;

    bool calibrated = true;

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

    void
    pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(),
                    "Received a newly published sensor message.");

        if (!this->calibrated) {
            RCLCPP_WARN(this->get_logger(),
                        "Calibration not done yet. Ignoring point cloud.");
            return;
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

        // Crop the point cloud

        // Convert pcl::PointCloud to octomap
        for (const auto &point : transformed_cloud->points) {
            current_tree->updateNode(
                octomap::point3d(point.x, point.y, point.z), true);
        }
        RCLCPP_INFO(this->get_logger(),
                    "Converted pcl point cloud to octomap::octree.\n");

        // Convert octomap to octomap_msgs::Octomap and publish
        octomap_msgs::msg::Octomap octomap_msg;
        octomap_msgs::fullMapToMsg(*current_tree, octomap_msg);
        octomap_publisher_->publish(octomap_msg);
        RCLCPP_INFO(this->get_logger(),
                    "Octomap_msg published to /kinect2_map/octomap.\n");
    }

    void
    saveOctreeCallback(const std_srvs::srv::Trigger::Request::SharedPtr request,
                       std_srvs::srv::Trigger::Response::SharedPtr response) {
        if (!this->calibrated) {
            response->success = false;
            response->message = "Calibration not done yet.";
            return;
        }

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
