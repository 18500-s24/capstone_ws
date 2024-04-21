#include <octomap/octomap.h>                /* octomap */
#include <octomap_msgs/conversions.h>       /* octomap_msgs::fullMapToMsg */
#include <rclcpp/rclcpp.hpp>                /* obvious */
#include <sensor_msgs/msg/point_cloud2.hpp> /* sensor_msgs::msg::PointCloud2 */
#include <std_msgs/msg/u_int8_multi_array.hpp> /* std_msgs::msg::UInt8MultiArray */
#include <std_srvs/srv/trigger.hpp>

#include <pcl/point_cloud.h>                 /* pcl::PointCloud */
#include <pcl/point_types.h>                 /* pcl::PointXYZ */
#include <pcl_conversions/pcl_conversions.h> /* pcl_conversions::fromROSMsg */

class KinectOctomapNode : public rclcpp::Node {
  public:
    KinectOctomapNode() : Node("kinect_octomap_node"), current_tree(0.01) {
        // Initialize publishers and subscribers
        point_cloud_subscriber_ =
            this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/kinect2/hd/points",
                rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(
                                rmw_qos_profile_sensor_data))
                    .best_effort(),
                std::bind(&KinectOctomapNode::pointCloudCallback, this,
                          std::placeholders::_1));
        octomap_publisher_ = this->create_publisher<octomap_msgs::msg::Octomap>(
            "/kinect2_map/octomap", 10);
        intarray_publisher_ =
            this->create_publisher<std_msgs::msg::UInt8MultiArray>(
                "/kinect2_map/intarray", 10);

        // Declare a parameter for storing the output path
        this->declare_parameter<std::string>("output_path", "~/tree.bt");

        // Service to save the octree
        save_service_ = this->create_service<std_srvs::srv::Trigger>(
            "save_octree",
            std::bind(&KinectOctomapNode::saveOctreeCallback, this,
                      std::placeholders::_1, std::placeholders::_2));
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        point_cloud_subscriber_;
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_publisher_;
    rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr
        intarray_publisher_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;

    octomap::OcTree current_tree;

    bool calibrated = false;

    void
    pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!this->calibrated)
            return;

        // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Convert pcl::PointCloud to octomap
        current_tree.clear();
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

        current_tree.writeBinary(
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
