#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class KinectOctomapNode : public rclcpp::Node {
  public:
    KinectOctomapNode() : Node("kinect_octomap_node") {
        // Initialize publishers and subscribers
        point_cloud_subscriber_ =
            this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/kinect2/hd/points",
                rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(
                                rmw_qos_profile_sensor_data))
                    .best_effort(),
                std::bind(&KinectOctomapNode::pointCloudCallback, this,
                          std::placeholders::_1));
        octomap_publisher_ =
            this->create_publisher<octomap_msgs::msg::Octomap>("octomap", 10);

        // Declare a parameter for storing the output path
        this->declare_parameter<std::string>("output_path",
                                             "~/capstone/tree.bt");
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        point_cloud_subscriber_;
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_publisher_;

    void
    pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Convert pcl::PointCloud to octomap
        octomap::OcTree tree(0.01); // resolution of 0.05 meters
        for (const auto &point : cloud->points) {
            tree.updateNode(octomap::point3d(point.x, point.y, point.z), true);
        }

        // Convert octomap to octomap_msgs::Octomap and publish
        octomap_msgs::msg::Octomap octomap_msg;
        octomap_msgs::fullMapToMsg(tree, octomap_msg);
        octomap_publisher_->publish(octomap_msg);

        // Dump the tree
        std::string output_path;
        this->get_parameter("output_path", output_path);
        tree.writeBinary(output_path);
        RCLCPP_INFO(this->get_logger(), "Saved OctoMap to %s",
                    output_path.c_str());
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KinectOctomapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
