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

#include <cmath>   /* sin, cos, sqrt, pow, abs, round */
#include <fstream> /* std::ifstream */
#include <math.h>  /* M_PI */
#include <string>  /* std::string */

static bool isClose(float a, float b) { return std::abs(a - b) < 0.0001; }

static float roundTo(float num, float precision) {
    return std::round(num / precision) * precision;
}

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

        // parameters
        this->declare_parameter<bool>("crop_scene", false);
        this->declare_parameter<bool>("prune_unreachable", false);
        this->declare_parameter<bool>("fill_back", false);
        this->declare_parameter<double>("octree_resolution", 0.01);
        this->declare_parameter<std::string>("octree_output_path", "~/tree.bt");
        this->declare_parameter<std::string>("octree_augmented_output_path",
                                             "~/tree_augmented.bt");
        this->declare_parameter<std::string>("intarray_output_path",
                                             "~/scene.txt");
        this->declare_parameter<double>("x_rotation", 240.0);
        this->declare_parameter<double>("y_rotation", 0.0);
        this->declare_parameter<double>("z_rotation", 0.0);
        this->declare_parameter<double>("x_translation", 0.0);
        this->declare_parameter<double>("y_translation", 0.0);
        this->declare_parameter<double>("z_translation", 0.0);
        this->declare_parameter<double>("scene_x_max", 1.27);      // in meters
        this->declare_parameter<double>("scene_y_max", 0.63);      // in meters
        this->declare_parameter<double>("scene_z_max", 0.63);      // in meters
        this->declare_parameter<double>("fill_y_threshold", 0.15); // in meters
        this->declare_parameter<double>("arm_base_x", 0.61);       // in meters
        this->declare_parameter<double>("arm_base_y", 0.0);        // in meters
        this->declare_parameter<double>("arm_base_z", 0.0);        // in meters
        this->declare_parameter<double>("arm_max_reach", 0.5);     // in meters
        this->declare_parameter<double>("start_x", 0.20);          // in meters
        this->declare_parameter<double>("start_y", 0.20);          // in meters
        this->declare_parameter<double>("start_z", 0.20);          // in meters
        this->declare_parameter<double>("end_x", 1.00);            // in meters
        this->declare_parameter<double>("end_y", 0.20);            // in meters
        this->declare_parameter<double>("end_z", 0.20);            // in meters

        // service to save the octree
        save_octree_service_ = this->create_service<std_srvs::srv::Trigger>(
            "save_octree",
            std::bind(&KinectOctomapNode::saveOctreeCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // service to save the intarray
        save_intarray_service_ = this->create_service<std_srvs::srv::Trigger>(
            "save_intarray",
            std::bind(&KinectOctomapNode::saveIntArrayCallback, this,
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

    // services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_octree_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_intarray_service_;

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
            this->get_parameter("arm_max_reach").as_double());
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

        RCLCPP_INFO(this->get_logger(),
                    "Pruned unreachable regions in octree.");
    }

    /**
     * @brief Fill the back of the objects in the octree
     */
    void fillBackOfObjects(octomap::OcTree *tree) {
        float resolution = this->current_tree->getResolution();
        float scene_x_min = 0;
        float scene_y_min = 0;
        float scene_z_min = 0;
        float scene_x_max =
            static_cast<float>(this->get_parameter("scene_x_max").as_double());
        float scene_y_max =
            static_cast<float>(this->get_parameter("scene_y_max").as_double());
        float scene_z_max =
            static_cast<float>(this->get_parameter("scene_z_max").as_double());
        float fill_y_threshold = static_cast<float>(
            this->get_parameter("fill_y_threshold").as_double());

        // for each voxel in the x-z plane with y >= y_threshold, if the
        // voxel is occupied, fill all the voxels behind it
        for (float x = scene_x_min; x < scene_x_max; x += resolution) {
            for (float z = scene_z_min; z < scene_z_max; z += resolution) {
                bool found_occupied = false;
                float y_occupied = 0.0;

                float x_center = x + (resolution / 2.0);
                float z_center = z + (resolution / 2.0);

                // find the first occupied voxel in the y direction
                for (float y = fill_y_threshold; y < scene_y_max;
                     y += resolution) {
                    float y_center = y + (resolution / 2.0);

                    octomap::OcTreeNode *node =
                        tree->search(x_center, y_center, z_center);

                    // if node is not null, then it is occupied
                    if (node != nullptr) {
                        found_occupied = true;
                        y_occupied = y;
                        break;
                    }
                }

                // if an occupied voxel is found, fill all the voxels behind it
                if (found_occupied) {
                    for (float y = y_occupied; y < scene_y_max;
                         y += resolution) {
                        float y_center = y + (resolution / 2.0);

                        tree->updateNode(
                            octomap::point3d(x_center, y_center, z_center),
                            true);
                    }
                }
            }
        }

        RCLCPP_INFO(this->get_logger(), "Filled back of objects in octree.");
    }

    void
    pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(),
                    "Received a newly published sensor message.");

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

        // Crop the point cloud to the scene dimensions
        if (this->get_parameter("crop_scene").as_bool()) {
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
        if (this->get_parameter("prune_unreachable").as_bool()) {
            pruneUnreachableRegions(current_tree);
        }

        // Mark the back of the objects as occupied (since the kinect cannot
        // see the back of the objects)
        if (this->get_parameter("fill_back").as_bool()) {
            fillBackOfObjects(current_tree);
        }
    }

    void
    saveOctreeCallback(const std_srvs::srv::Trigger::Request::SharedPtr request,
                       std_srvs::srv::Trigger::Response::SharedPtr response) {
        std::ifstream tree_file;
        tree_file.open(this->get_parameter("octree_output_path").as_string());

        // write the octree
        current_tree->writeBinary(
            this->get_parameter("octree_output_path").as_string());

        // write the "augmented" octree with start and end voxels
        float scene_x_min = 0.0;
        float scene_y_min = 0.0;
        float scene_z_min = 0.0;
        float scene_x_max =
            static_cast<float>(this->get_parameter("scene_x_max").as_double());
        float scene_y_max =
            static_cast<float>(this->get_parameter("scene_y_max").as_double());
        float scene_z_max =
            static_cast<float>(this->get_parameter("scene_z_max").as_double());
        float start_x =
            static_cast<float>(this->get_parameter("start_x").as_double());
        float start_y =
            static_cast<float>(this->get_parameter("start_y").as_double());
        float start_z =
            static_cast<float>(this->get_parameter("start_z").as_double());
        float end_x =
            static_cast<float>(this->get_parameter("end_x").as_double());
        float end_y =
            static_cast<float>(this->get_parameter("end_y").as_double());
        float end_z =
            static_cast<float>(this->get_parameter("end_z").as_double());

        // add the start and end voxels to the octree
        current_tree->updateNode(octomap::point3d(start_x, start_y, start_z),
                                 true);
        current_tree->updateNode(octomap::point3d(end_x, end_y, end_z), true);

        // write the "augmented" octree
        current_tree->writeBinary(
            this->get_parameter("octree_augmented_output_path").as_string());

        // remove the start and end voxels from the octree
        current_tree->deleteNode(octomap::point3d(start_x, start_y, start_z));
        current_tree->deleteNode(octomap::point3d(end_x, end_y, end_z));

        response->success = true;
        response->message = "Octree saved successfully.";

        RCLCPP_INFO(this->get_logger(), "Octree saved successfully.");
    }

    void saveIntArrayCallback(
        const std_srvs::srv::Trigger::Request::SharedPtr request,
        std_srvs::srv::Trigger::Response::SharedPtr response) {
        RCLCPP_INFO(this->get_logger(), "Saving IntArray.");

        float resolution = this->current_tree->getResolution();
        float scene_x_min = 0;
        float scene_y_min = 0;
        float scene_z_min = 0;
        float scene_x_max =
            static_cast<float>(this->get_parameter("scene_x_max").as_double());
        float scene_y_max =
            static_cast<float>(this->get_parameter("scene_y_max").as_double());
        float scene_z_max =
            static_cast<float>(this->get_parameter("scene_z_max").as_double());

        float start_x =
            static_cast<float>(this->get_parameter("start_x").as_double());
        float start_y =
            static_cast<float>(this->get_parameter("start_y").as_double());
        float start_z =
            static_cast<float>(this->get_parameter("start_z").as_double());
        size_t start_x_ind =
            static_cast<size_t>((start_x - scene_x_min) / resolution);
        size_t start_y_ind =
            static_cast<size_t>((start_y - scene_y_min) / resolution);
        size_t start_z_ind =
            static_cast<size_t>((start_z - scene_z_min) / resolution);

        float end_x =
            static_cast<float>(this->get_parameter("end_x").as_double());
        float end_y =
            static_cast<float>(this->get_parameter("end_y").as_double());
        float end_z =
            static_cast<float>(this->get_parameter("end_z").as_double());
        size_t end_x_ind =
            static_cast<size_t>((end_x - scene_x_min) / resolution);
        size_t end_y_ind =
            static_cast<size_t>((end_y - scene_y_min) / resolution);
        size_t end_z_ind =
            static_cast<size_t>((end_z - scene_z_min) / resolution);

        // iterate through the scene and write to the output
        // unoccupied voxels are written as 0 (raw byte)
        // occupied voxels are written as 1 (raw byte)
        // start and end voxels are written as 2 (raw byte)
        size_t x_dim =
            static_cast<size_t>((scene_x_max - scene_x_min) / resolution) + 1;
        size_t y_dim =
            static_cast<size_t>((scene_y_max - scene_y_min) / resolution) + 1;
        size_t z_dim =
            static_cast<size_t>((scene_z_max - scene_z_min) / resolution) + 1;
        uint8_t out[x_dim][y_dim][z_dim] = {0};

        // get all the occupied voxels
        // for each occupied voxel, determine if it is a start or end voxel
        // if so, write 2 to the output
        // else, write 1 to the output
        for (auto it = current_tree->begin(); it != current_tree->end(); ++it) {
            octomap::point3d center = it.getCoordinate();
            float x = center.x();
            float y = center.y();
            float z = center.z();

            size_t x_ind = static_cast<size_t>((x - scene_x_min) / resolution);
            size_t y_ind = static_cast<size_t>((y - scene_y_min) / resolution);
            size_t z_ind = static_cast<size_t>((z - scene_z_min) / resolution);

            out[x_ind][y_ind][z_ind] = 1;
        }

        if (out[start_x_ind][start_y_ind][start_z_ind] != 0) {
            RCLCPP_ERROR(this->get_logger(),
                         "Start voxel is already occupied.");
        } else if (out[end_x_ind][end_y_ind][end_z_ind] != 0) {
            RCLCPP_ERROR(this->get_logger(), "End voxel is already occupied.");
        }

        out[start_x_ind][start_y_ind][start_z_ind] = 2;
        out[end_x_ind][end_y_ind][end_z_ind] = 2;

        std::ofstream intarray_output_file;
        intarray_output_file.open(
            this->get_parameter("intarray_output_path").as_string(),
            std::ios::out | std::ios::binary);
        intarray_output_file.write((char *)out, sizeof(out));
        intarray_output_file.close();

        response->success = true;
        response->message = "IntArray saved successfully.";

        RCLCPP_INFO(this->get_logger(), "IntArray saved successfully.");
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KinectOctomapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
