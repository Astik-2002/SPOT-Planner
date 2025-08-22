#include <iostream>
#include <math.h>
#include <random>
#include <Eigen/Eigen>
#include "kdtree_dynamic.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp>
#include <unordered_set>
#include "datatype_dynamic.h"
#include "custom_hash.hpp"
#include <memory.h>
#include <unordered_map>
#include <tuple>
#include <functional>
class safeRegionRrtStarDynamic
{
	private:
		pcl::search::KdTree<pcl::PointXYZ> kdtreeForMap;
		pcl::search::KdTree<pcl::PointXYZ> kdtreeForCollisionPred;

		pcl::search::KdTree<pcl::PointXYZ> kdtreeAddMap;
		pcl::search::KdTree<pcl::PointXYZ> kdtreeDelMap;
		pcl::PointCloud<pcl::PointXYZ> CloudIn;
		
		kdtree * kdTree_; // dynamic light-weight Kd-tree, for organizing the nodes in the exploring tree

		vector<int>     pointIdxRadiusSearch;
		vector<float>   pointRadiusSquaredDistance;        
		
		// All nodes,  nodes reach the target
		vector<NodePtr_dynamic> NodeList, EndList;
		std::unordered_map<Node_dynamic*, NodePtr_dynamic> node_raw_to_shared_map;
		// all nodes in the current path, for easy usage in root re-decleration 
		vector<NodePtr_dynamic> PathList, invalidSet;

		vector<Eigen::Vector4d> selectedNodeList, sampleList;
		// record the current best node which can reach the target and has the lowest path cost
		NodePtr_dynamic best_end_ptr;

		// record the root of the rapidly exploring tree
		NodePtr_dynamic root_node;

		// Update your temporal_grid definition
		// std::unordered_map<int, std::unordered_map<std::tuple<int, int, int>, std::vector<Eigen::Vector3d>, TupleHash>> temporal_grid;
		std::map<int, std::unordered_map<std::tuple<int,int,int>, std::vector<Eigen::Vector3d>, TupleHash>> temporal_grid;
		// Parameters
		double time_resolution = 0.2;  // Time bin size (seconds)
		double cell_size = 2.0;

		// start point,   target point,  centroid point of the ellipsoide sampling region
		Eigen::Vector3d start_pt, inform_centroid, pcd_origin, end_pt;
		Eigen::Vector4d commit_root;
		// ctrl the size of the trash nodes cach, once it's full, remove them and rebuild the kd-tree
		int cach_size = 100; 

		// maximum allowed samples
		int max_samples = 30000000;

        double delta_t = 1.0;
		int num_frames = 10;
		double horizon_time = 1000.0;
		double weightT = 0.3;
		double PCDstartTime = 0.0;
		std::vector<pcl::search::KdTree<pcl::PointXYZ>> kdtreeForMapList;
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> dynamic_points;
		// used for basic of sampling-based path finding
		double x_l, x_h, y_l, y_h, z_l, z_h, bias_l, bias_h, inlier_ratio, goal_ratio;
		
		// used for range query and search strategy
		double safety_margin, max_radius, search_margin, sample_range;
		
		// used for the informed sampling strategy
		double min_distance, best_distance, elli_l, elli_s, ctheta, stheta;   
		
		double current_yaw, expected_time, average_vel, max_vel;
		// FLAGs
		bool inform_status, path_exist_status, global_navi_status, uncertanity;

		Eigen::MatrixXd Path;
		Eigen::VectorXd Radius;
		double h_fov;
		double v_fov;
		int pcd_sensor_nums;
		random_device rd;
		default_random_engine eng;
		uniform_real_distribution<double>  rand_rho = uniform_real_distribution<double>(0.0, 1.0);  // random distribution for generating samples inside a unit circle
		uniform_real_distribution<double> rand_idx = uniform_real_distribution<double>(0, 1000);
		uniform_real_distribution<double>  rand_phi = uniform_real_distribution<double>(0.0, 2 * M_PI);
		uniform_real_distribution<double>  rand_x,    rand_y,    rand_z,   rand_bias; // basic random distributions for generating samples, in all feasible regions
		uniform_real_distribution<double>  rand_x_in, rand_y_in, rand_z_in; // random distribution, especially for generating samples inside the local map's boundary

    /* ---------- uniformly sampling a 3D ellipsoid ---------- */
    uniform_real_distribution<double> rand_u, rand_v;  // U ~ (0,1)
	std::vector<double> time_samples;
    Eigen::Vector3d translation_inf;
    Eigen::Matrix3d rotation_inf;

  public:
		safeRegionRrtStarDynamic( );
        ~safeRegionRrtStarDynamic();
		
		/* set-up functions */
		void reset();
		void setParam( double safety_margin_, double search_margin_, double deltat, double sample_range_, double h_fov_, double v_fov , bool uncertanity);
		void setInputStatic(pcl::PointCloud<pcl::PointXYZ> CloudIn);
		void setInputDynamic(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &dynamic_points,
		const Eigen::Vector3d &origin,
		double start_time);
		void buildTemporalGrid(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& dynamic_points);
		double queryDynamicObstacles(const Eigen::Vector4d& query_pt); 
		void setInputforCollision(pcl::PointCloud<pcl::PointXYZ> CloudIn);
		void setPt( Eigen::Vector3d startPt, Eigen::Vector3d endPt, double xl, double xh, double yl, double yh, double zl, double zh,
					double local_range, int max_iter, double sample_portion, double goal_portion, double yaw, double _average_vel, double _max_vel, double weight_t);

		void setStartPt( Eigen::Vector3d startPt, Eigen::Vector3d endPt);
		void checkingRadius(NodePtr_dynamic node);
		/*  commit local target and move tree root  */
		void resetRoot(Eigen::Vector4d & commitTarget);
		void resetRoot(int root_index);
		void solutionUpdate(double cost_reduction, Eigen::Vector4d target);

		/* main function entries */
		void SafeRegionExpansion( double time_limit, double root_node_time);
		void SafeRegionRefine   ( double time_limit);
		void SafeRegionEvaluate ( double time_limit);

		/* operations on the tree */
		void treeRepair(double time_limit, vector<pair<Eigen::Vector4d, double>>& node_list);
		void treePrune( NodePtr_dynamic newPtr );
		void treeRewire( NodePtr_dynamic node_new_ptr, NodePtr_dynamic node_nearst_ptr );
		void clearBranchW(NodePtr_dynamic node_delete); // weak branch cut:    clear branches while avoid deleting the nodes on the best path
		void clearBranchS(NodePtr_dynamic node_delete); // strong branch cut:  clear all nodes on a branch
		void updateHeuristicRegion( NodePtr_dynamic update_end_node );
		void recordNode(NodePtr_dynamic new_node);		
		void removeInvalid();
		void removeInvalid4d();
		void treeDestruct();
		void tracePath();

		/* utility functions */
        inline double getDis(const NodePtr_dynamic node1, const NodePtr_dynamic node2);
		inline double getDis(const NodePtr_dynamic node1, const Eigen::Vector3d & pt);
		inline double getDis(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2);

		inline double getDis4d(const NodePtr_dynamic node1, const NodePtr_dynamic node2);
		inline double getDis4d(const NodePtr_dynamic node1, const Eigen::Vector4d & pt);
		inline double getDis4d(const Eigen::Vector4d & p1, const Eigen::Vector4d & p2);
		inline std::pair<double, double> getDisMetric(const NodePtr_dynamic node_p, const NodePtr_dynamic node_c);
		inline std::pair<double, double> getDisMetric(const Eigen::Vector4d node_p,  const Eigen::Vector4d node_c);

		inline Eigen::Vector3d genSample();
		inline Eigen::Vector4d genSample4d();

		Eigen::Vector4d getRootCoords();
		inline double radiusSearch(Eigen::Vector4d & pt);
		inline double radiusSearchCollisionPred(Eigen::Vector3d & pt);
		inline NodePtr_dynamic genNewNode( Eigen::Vector4d & pt, NodePtr_dynamic node_nearst_ptr );
		inline NodePtr_dynamic findNearestVertex( Eigen::Vector3d & pt );
		inline NodePtr_dynamic findNearestVertex4d( Eigen::Vector4d & pt );

		inline double checkRadius(Eigen::Vector4d & pt);
		inline bool checkValidEnd(NodePtr_dynamic endPtr);
		inline bool checkEnd( NodePtr_dynamic ptr );
		inline int  checkNodeUpdate  ( double new_radius, double old_radius);
		inline int  checkNodeRelation( NodePtr_dynamic node_1, NodePtr_dynamic node_2 );
		inline bool isSuccessor(NodePtr_dynamic curPtr, NodePtr_dynamic nearPtr);
		bool checkTrajPtCol(Eigen::Vector4d & pt);
		Eigen::Vector3d computeNoiseStd(const Eigen::Vector3d& point);
		bool isInFOV(const Eigen::Vector3d& pt);

		/* data return */
		pair<Eigen::MatrixXd, Eigen::VectorXd> getPath()
		{
			return make_pair(Path, Radius);
		};
		
		vector<NodePtr_dynamic> getTree()
		{
			return NodeList;
		};

		vector<NodePtr_dynamic> getPathList()
		{
			return PathList;
		};
		
		vector<Eigen::Vector4d> getSelectedNodeList()
		{
			return selectedNodeList;
		};

		void updateTimeSamples() 
		{
			time_samples.clear();
			if (delta_t > 0 && expected_time > 0) {
				for (double t = 0; t <= expected_time; t += delta_t) {
					time_samples.push_back(t);
				}
			}
		}
		
		vector<Eigen::Vector4d> getSampleList()
		{
			return sampleList;
		};
		
		bool getPathExistStatus()
		{
			return path_exist_status;
		};

		bool getGlobalNaviStatus()
		{
			return global_navi_status;
		};
};