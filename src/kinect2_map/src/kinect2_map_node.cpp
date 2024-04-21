#include <octomap/octomap.h>                /* octomap */
#include <octomap_msgs/conversions.h>       /* octomap_msgs::fullMapToMsg */
#include <rclcpp/rclcpp.hpp>                /* obvious */
#include <sensor_msgs/msg/point_cloud2.hpp> /* sensor_msgs::msg::PointCloud2 */
#include <std_msgs/msg/u_int8_multi_array.hpp> /* std_msgs::msg::UInt8MultiArray */
#include <std_srvs/srv/trigger.hpp>            /* std_srvs::srv::Trigger */

#include <pcl/point_cloud.h>                 /* pcl::PointCloud */
#include <pcl/point_types.h>                 /* pcl::PointXYZ */
#include <pcl_conversions/pcl_conversions.h> /* pcl_conversions::fromROSMsg */

#include <fstream> /* std::ifstream */

class KinectOctomapNode : public rclcpp::Node {
  public:
    KinectOctomapNode() : Node("kinect_octomap_node"), current_tree(0.01) {
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

        // parameter for storing the output path
        this->declare_parameter<std::string>("output_path", "~/tree.bt");

        // service to save the octree
        save_octree_service_ = this->create_service<std_srvs::srv::Trigger>(
            "save_octree",
            std::bind(&KinectOctomapNode::saveOctreeCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // service to calibrate the base
        calibrate_base_service_ = this->create_service<std_srvs::srv::Trigger>(
            "calibrate_base",
            std::bind(&KinectOctomapNode::calibrateBaseCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // service to calibrate the xmin ymin
        calibrate_xmin_ymin_service_ =
            this->create_service<std_srvs::srv::Trigger>(
                "calibrate_xmin_ymin",
                std::bind(&KinectOctomapNode::calibrateXminYminCallback, this,
                          std::placeholders::_1, std::placeholders::_2));

        // service to calibrate the xmax ymin
        calibrate_xmax_ymin_service_ =
            this->create_service<std_srvs::srv::Trigger>(
                "calibrate_xmax_ymin",
                std::bind(&KinectOctomapNode::calibrateXmaxYminCallback, this,
                          std::placeholders::_1, std::placeholders::_2));

        // service to calibrate the xmin ymax
        calibrate_xmin_ymax_service_ =
            this->create_service<std_srvs::srv::Trigger>(
                "calibrate_xmin_ymax",
                std::bind(&KinectOctomapNode::calibrateXminYmaxCallback, this,
                          std::placeholders::_1, std::placeholders::_2));

        // service to calibrate the xmax ymax
        calibrate_xmax_ymax_service_ =
            this->create_service<std_srvs::srv::Trigger>(
                "calibrate_xmax_ymax",
                std::bind(&KinectOctomapNode::calibrateXmaxYmaxCallback, this,
                          std::placeholders::_1, std::placeholders::_2));
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
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
        calibrate_xmin_ymin_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
        calibrate_xmax_ymin_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
        calibrate_xmin_ymax_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
        calibrate_xmax_ymax_service_;

    octomap::OcTree current_tree;

    bool calibrated = false;
    bool calibrated_base = false;
    bool calibrated_xmin_ymin = false;
    bool calibrated_xmax_ymin = false;
    bool calibrated_xmin_ymax = false;
    bool calibrated_xmax_ymax = false;

    void
    pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!this->calibrated)
            return;

        // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Convert pcl::PointCloud to octomap
        for (const auto &point : cloud->points) {
            current_tree.updateNode(octomap::point3d(point.x, point.y, point.z),
                                    true);
        }

        // Convert octomap to octomap_msgs::Octomap and publish
        octomap_msgs::msg::Octomap octomap_msg;
        octomap_msgs::fullMapToMsg(current_tree, octomap_msg);
        octomap_publisher_->publish(octomap_msg);
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

        current_tree.writeBinary(
            this->get_parameter("output_path").as_string());
        response->success = true;
        response->message = "Octree saved successfully.";
    }

    void updateCalibrationStatus() {
        this->calibrated =
            this->calibrated_base && this->calibrated_xmin_ymin &&
            this->calibrated_xmax_ymin && this->calibrated_xmin_ymax &&
            this->calibrated_xmax_ymax;
    }

    void calibrateBaseCallback(
        const std_srvs::srv::Trigger::Request::SharedPtr request,
        std_srvs::srv::Trigger::Response::SharedPtr response) {
        this->calibrated_base = true;
        this->updateCalibrationStatus();
        response->success = true;
        response->message = "Base calibrated successfully.";
    }

    void calibrateXminYminCallback(
        const std_srvs::srv::Trigger::Request::SharedPtr request,
        std_srvs::srv::Trigger::Response::SharedPtr response) {
        this->calibrated_xmin_ymin = true;
        this->updateCalibrationStatus();
        response->success = true;
        response->message = "Xmin Ymin calibrated successfully.";
    }

    void calibrateXmaxYminCallback(
        const std_srvs::srv::Trigger::Request::SharedPtr request,
        std_srvs::srv::Trigger::Response::SharedPtr response) {
        this->calibrated_xmax_ymin = true;
        this->updateCalibrationStatus();
        response->success = true;
        response->message = "Xmax Ymin calibrated successfully.";
    }

    void calibrateXminYmaxCallback(
        const std_srvs::srv::Trigger::Request::SharedPtr request,
        std_srvs::srv::Trigger::Response::SharedPtr response) {
        this->calibrated_xmin_ymax = true;
        this->updateCalibrationStatus();
        response->success = true;
        response->message = "Xmin Ymax calibrated successfully.";
    }

    void calibrateXmaxYmaxCallback(
        const std_srvs::srv::Trigger::Request::SharedPtr request,
        std_srvs::srv::Trigger::Response::SharedPtr response) {
        this->calibrated_xmax_ymax = true;
        this->updateCalibrationStatus();
        response->success = true;
        response->message = "Xmax Ymax calibrated successfully.";
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KinectOctomapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
