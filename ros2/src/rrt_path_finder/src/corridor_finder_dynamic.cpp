#include "rrt_path_finder/corridor_finder_dynamic.h"
#include <algorithm>
#include <chrono>
using namespace Eigen;
using namespace std;

safeRegionRrtStarDynamic::safeRegionRrtStarDynamic( ){ 
      cach_size  = 100;
}

safeRegionRrtStarDynamic::~safeRegionRrtStarDynamic(){ }

Eigen::Vector3d safeRegionRrtStarDynamic::computeNoiseStd(const Eigen::Vector3d& point) {
    double x = point.x();
    // Compute axial noise (σz) using the formula from the paper
    double sigma_x = 0.001063 + 0.0007278 * x + 0.003949 * x * x;

    // Compute lateral noise (σx, σy)
    double sigma_y = 0.04;
    double sigma_z = 0.04;

    return Eigen::Vector3d(sigma_x, sigma_y, sigma_z);
}

void safeRegionRrtStarDynamic::setParam( double safety_margin_, double search_margin_, double max_radius_, double sample_range_, double h_fov_, double v_fov_, bool uncertanity_ )
{   
    std::cout<<"set param called"<<std::endl;
    safety_margin = safety_margin_;
    search_margin = search_margin_;
    max_radius    = max_radius_;
    sample_range  = sample_range_;
    h_fov = h_fov_;
    v_fov = v_fov_;\
    uncertanity = uncertanity_;
}

void safeRegionRrtStarDynamic::reset()
{     
    treeDestruct();
    
    NodeList.clear();
    EndList.clear();
    invalidSet.clear();
    PathList.clear();

    best_end_ptr = new Node_dynamic();
    root_node    = new Node_dynamic();

    path_exist_status  = true;
    inform_status      = false;
    global_navi_status = false;
    best_distance      = INFINITY;  
}

void safeRegionRrtStarDynamic::setStartPt( Vector3d startPt, Vector3d endPt, double averageVel)
{
    start_pt = startPt;
    end_pt.head<3>()   = endPt; 
    average_vel = averageVel;
    end_pt[3] = getDis(start_pt, end_pt.head<3>()) / average_vel;
    rand_x_in = uniform_real_distribution<double>(start_pt(0) - sample_range, start_pt(0) + sample_range);
    rand_y_in = uniform_real_distribution<double>(start_pt(1) - sample_range, start_pt(1) + sample_range);
}

void safeRegionRrtStarDynamic::setPt( Vector3d startPt, Vector3d endPt, double xl, double xh, double yl, double yh, double zl, double zh,
                           double local_range, int max_iter, double sample_portion, double goal_portion, double yaw, double _average_vel, double _max_vel, double _horizon_time, double _delta_t)
{
    start_pt = startPt;
    end_pt.head<3>() = endPt; 
    
    x_l = xl; x_h = xh;
    y_l = yl; y_h = yh;
    z_l = zl; z_h = zh;

    bias_l = 0.0; bias_h = 1.0;
    
    eng = default_random_engine(rd()) ;
    rand_x    = uniform_real_distribution<double>(x_l, x_h);
    rand_y    = uniform_real_distribution<double>(y_l, y_h);
    rand_z = rand_z_in = uniform_real_distribution<double>(z_l, z_h);
    rand_bias = uniform_real_distribution<double>(bias_l, bias_h);

    rand_x_in = uniform_real_distribution<double>(start_pt(0) - sample_range, start_pt(0) + sample_range);
    rand_y_in = uniform_real_distribution<double>(start_pt(1) - sample_range, start_pt(1) + sample_range);

    min_distance = sqrt( pow(start_pt(0) - end_pt(0), 2) + pow(start_pt(1) - end_pt(1), 2) + pow(start_pt(2) - end_pt(2), 2) );

    /* ---------- update ellipsoid ellipsoid transformation ---------- */
    /* 3D ellipsoid transformation */
    translation_inf = (start_pt + end_pt.head<3>()) / 2.0;

    Eigen::Vector3d xtf, ytf, ztf, downward(0, 0, -1);
    xtf = (end_pt.head<3>() - translation_inf).normalized();
    ytf = xtf.cross(downward).normalized();
    ztf = xtf.cross(ytf);

    rotation_inf.col(0) = xtf;
    rotation_inf.col(1) = ytf;
    rotation_inf.col(2) = ztf;

    sample_range = local_range;
    max_samples  = max_iter;
    inlier_ratio = sample_portion;
    goal_ratio   = goal_portion;
    current_yaw = yaw;
    average_vel = _average_vel;
    max_vel = _max_vel;
    expected_time = getDis(start_pt, end_pt.head<3>()) / average_vel;
    end_pt[3] = expected_time;
    horizon_time = _horizon_time;
    num_frames = static_cast<int>(horizon_time / delta_t);
    int num_possible_t = static_cast<int>(expected_time / delta_t);
    rand_idx = std::uniform_int_distribution<int>(0, num_possible_t-1);
    delta_t = _delta_t;
    time_samples.clear();
    for (int i = 0; i < num_possible_t; ++i)
    {
        time_samples.push_back(i * delta_t);
    }
}

void safeRegionRrtStarDynamic::setInput(pcl::PointCloud<pcl::PointXYZ> cloud_in, Eigen::Vector3d origin)
{     
    kdtreeForMap.setInputCloud( cloud_in.makeShared() );  
    CloudIn = cloud_in;
    pcd_origin = origin;
}

// void safeRegionRrtStarDynamic::setInputDynamic(pcl::PointCloud<pcl::PointXYZ> cloud_in, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> dynamic_points, Eigen::Vector3d origin, int pcd_idx)
// {   
//     // pcd_idx is the index of the first pointcloud in the sequence i.e. ith pointcloud received
//     // position, velocity
//     kdtreeForMapList.clear();
//     kdtreeForMapList.resize(num_frames);
//     std::vector<pcl::PointCloud<pcl::PointXYZ>> vec_pcds(num_frames);
//     pcd_sensor_nums = pcd_idx;
//     std::cout<<"[dynamic input debug] pcd_sensor_nums: "<<pcd_sensor_nums<<std::endl;
//     // Pre-allocate memory assuming worst-case: all dynamic + all static per cloud
//     size_t dynamic_size = dynamic_points.size();
//     size_t static_size = cloud_in.size();

//     for (auto& pcd : vec_pcds)
//     {
//         pcd.points.reserve(dynamic_size + static_size);
//     }

//     // Single loop over dynamic points
//     for (const auto& [position, velocity] : dynamic_points)
//     {
//         Eigen::Vector3d vdt = delta_t * velocity;

//         Eigen::Vector3d p = position;
        
//         for (int i = 0; i < num_frames; ++i)
//         {
//             p += vdt;  // incrementally propagate to avoid recomputation
//             vec_pcds[i].points.emplace_back(p.x(), p.y(), p.z());
//         }
//     }

//     // More efficient: append static points to all vec_pcds[i]
//     for (int i = 0; i < num_frames; ++i)
//     {
//         vec_pcds[i].points.insert(vec_pcds[i].points.end(), cloud_in.points.begin(), cloud_in.points.end());
//         kdtreeForMapList[i].setInputCloud(vec_pcds[i].makeShared());
//     }
//     pcd_origin = origin;
// }

void safeRegionRrtStarDynamic::setInputDynamic(
    const pcl::PointCloud<pcl::PointXYZ>& cloud_in,
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& dynamic_points,
    const Eigen::Vector3d& origin,
    int pcd_idx)
{
    // Store pointcloud index and origin for use in planning
    pcd_sensor_nums = pcd_idx;
    pcd_origin = origin;

    // std::cout << "[dynamic input debug] pcd_sensor_nums: " << pcd_sensor_nums << std::endl;

    // Prepare KD-Trees and pointclouds for all timesteps
    kdtreeForMapList.clear();
    kdtreeForMapList.resize(num_frames);

    std::vector<pcl::PointCloud<pcl::PointXYZ>> vec_pcds(num_frames);

    const size_t dynamic_size = dynamic_points.size();
    const size_t static_size = cloud_in.size();
    const size_t total_estimated_size = dynamic_size + static_size;

    // Reserve total space to avoid reallocations
    for (auto& pcd : vec_pcds)
    {
        pcd.points.reserve(total_estimated_size);
    }

    // === Step 1: Add dynamic obstacle predictions to future pointclouds ===
    for (const auto& [position, velocity] : dynamic_points)
    {
        for (int i = 0; i < num_frames; ++i)
        {
            double t = (i + 1) * delta_t;  // future timestep
            Eigen::Vector3d future_pos = position + t * velocity;
            vec_pcds[i].points.emplace_back(future_pos.x(), future_pos.y(), future_pos.z());
        }
    }

    // === Step 2: Append static points to each timestep cloud ===
    // Note: static points are the same for all timesteps
    for (int i = 0; i < num_frames; ++i)
    {
        vec_pcds[i].points.insert(
            vec_pcds[i].points.end(),
            cloud_in.points.begin(),
            cloud_in.points.end()
        );
    }

    // === Step 3: Build KD-trees ===
    for (int i = 0; i < num_frames; ++i)
    {
        kdtreeForMapList[i].setInputCloud(vec_pcds[i].makeShared());
    }
}

inline double safeRegionRrtStarDynamic::getDis(const NodePtr_dynamic node1, const NodePtr_dynamic node2){
      return sqrt(pow(node1->coord(0) - node2->coord(0), 2) + pow(node1->coord(1) - node2->coord(1), 2) + pow(node1->coord(2) - node2->coord(2), 2) );
}

inline double safeRegionRrtStarDynamic::getDis(const NodePtr_dynamic node1, const Vector3d & pt){
      return sqrt(pow(node1->coord(0) - pt(0), 2) + pow(node1->coord(1) - pt(1), 2) + pow(node1->coord(2) - pt(2), 2) );
}

inline double safeRegionRrtStarDynamic::getDis(const Vector3d & p1, const Vector3d & p2){
      return sqrt(pow(p1(0) - p2(0), 2) + pow(p1(1) - p2(1), 2) + pow(p1(2) - p2(2), 2) );
}

inline double safeRegionRrtStarDynamic::getDis4d(const NodePtr_dynamic node1, const NodePtr_dynamic node2){
      return sqrt(pow(node1->coord(0) - node2->coord(0), 2) + pow(node1->coord(1) - node2->coord(1), 2) + pow(node1->coord(2) - node2->coord(2), 2) + pow(node1->coord(3) - node2->coord(3), 2) );
}

inline double safeRegionRrtStarDynamic::getDis4d(const NodePtr_dynamic node1, const Vector4d & pt){
      return sqrt(pow(node1->coord(0) - pt(0), 2) + pow(node1->coord(1) - pt(1), 2) + pow(node1->coord(2) - pt(2), 2) + pow(node1->coord(3) - pt(3), 2) );
}

inline double safeRegionRrtStarDynamic::getDis4d(const Vector4d & p1, const Vector4d & p2){
      return sqrt(pow(p1(0) - p2(0), 2) + pow(p1(1) - p2(1), 2) + pow(p1(2) - p2(2), 2) + pow(p1(3) - p2(3), 2) );
}

inline double safeRegionRrtStarDynamic::radiusSearch( Vector4d & search_Pt)
{     
    //    return max_radius - search_margin;

    pcl::PointXYZ searchPoint;
    searchPoint.x = search_Pt(0);
    searchPoint.y = search_Pt(1);
    searchPoint.z = search_Pt(2);
    double time_point = search_Pt(3);
    pointIdxRadiusSearch.clear();
    pointRadiusSquaredDistance.clear();
    double start_time = pcd_sensor_nums*delta_t; // time of the latest sensor measurement
    // std::cout<<"[radius search debug] pcd_sensor_nums: "<<pcd_sensor_nums<<std::endl;
    double prediction_end_time = (pcd_sensor_nums + num_frames)*delta_t;
    // std::cout<<"[radius search debug0]"<<std::endl;

    if(time_point > prediction_end_time)
    {
        // std::cout<<"[radius search debug 01]"<<std::endl;
        // std::cout << "[Error] Search point time exceeds prediction end time in radiusSearch()" << std::endl;
        return max_radius - search_margin;
    }
    
    if(time_point < start_time)
    {
        // std::cout<<"[radius search debug 02]"<<std::endl;
        // std::cout << "[Error] Search point time is before the start time in radiusSearch()" << std::endl;
        return -1.00;
    }
    else
    {
        // std::cout<<"[radius search debug 03]"<<std::endl;

        int pcd_idx = static_cast<int>((time_point - start_time) / delta_t);
        // std::cout<<"[radius search debug1]"<<std::endl;
        // std::cout<<"[radius search debug] time_point: "<<time_point<<": start_time: "<<start_time<<": pcd_idx: "<<pcd_idx<<std::endl;
        if (pcd_idx < 0 || pcd_idx >= num_frames) {
            // std::cout << "[Error] Invalid pcd_idx in radiusSearch()" << std::endl;
            return -1.00;
        }
        // std::cout<<"[radius search debug2]"<<std::endl;

        kdtreeForMapList[pcd_idx].nearestKSearch(searchPoint, 1, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        double radius = sqrt(pointRadiusSquaredDistance[0]) - search_margin;
        // std::cout<<"[debug] radius found successfully: "<<radius<<std::endl;
        return min(radius, double(max_radius));
    }
}

// void safeRegionRrtStarDynamic::clearBranchW(NodePtr_dynamic node_delete) // Weak branch cut: if a child of a node is on the current best path, keep the child
// {   // std::cout<<"clear branch seg1"<<std::endl;
//     if (!node_delete) {
//         std::cerr << "[Error] Node to delete is null in clearBranchW()" << std::endl;
//         return;
//     }
//     // std::cout<<"clear branch seg2, nxt-ptr size: "<<node_delete->nxtNode_ptr.size()<<std::endl;
    
//     if (!node_delete->nxtNode_ptr.empty())
//     {
//         for (auto NodePtr_dynamic : node_delete->nxtNode_ptr) 
//         {
//             if (!NodePtr_dynamic->best) {
//                 // std::cout<<"clear branch seg5"<<std::endl;
//                 if (NodePtr_dynamic->valid) {
//                     // std::cout<<"clear branch seg6"<<std::endl;
//                     invalidSet.push_back(NodePtr_dynamic);
//                 }
//                 // std::cout<<"clear branch seg7"<<std::endl;
//                 NodePtr_dynamic->valid = false;
//                 clearBranchW(NodePtr_dynamic);
//             }
//         }
//     }
// }

// void safeRegionRrtStarDynamic::clearBranchS(NodePtr_dynamic node_delete) // Strong branch cut: no matter how, cut all nodes in this branch
// {     
//     for( auto NodePtr_dynamic: node_delete->nxtNode_ptr ){
//         if( NodePtr_dynamic->valid)
//             invalidSet.push_back(NodePtr_dynamic);
//             NodePtr_dynamic->valid = false;
//             clearBranchS(NodePtr_dynamic);
//     }
// }

void safeRegionRrtStarDynamic::clearBranchW(NodePtr_dynamic node_delete) {
    if (!node_delete) {
        std::cerr << "[Error] Node to delete is null in clearBranchW()" << std::endl;
        return;
    }
    
    // Prevent processing already invalid nodes
    if (!node_delete->valid) return;

    for (auto child : node_delete->nxtNode_ptr) {
        if (!child || !child->valid) continue;  // Skip null/invalid
        
        if (!child->best) {  // Only clear non-best-path nodes
            if (std::find(invalidSet.begin(), invalidSet.end(), child) == invalidSet.end()) {
                invalidSet.push_back(child);
            }
            child->valid = false;
            clearBranchW(child);  // Recurse
        }
    }
}

void safeRegionRrtStarDynamic::clearBranchS(NodePtr_dynamic node_delete) {
    if (!node_delete) {
        std::cerr << "[Error] Node to delete is null in clearBranchS()" << std::endl;
        return;
    }
    
    // Prevent processing already invalid nodes
    if (!node_delete->valid) return;

    for (auto child : node_delete->nxtNode_ptr) {
        if (!child) continue;  // Skip null
        
        if (child->valid) {
            if (std::find(invalidSet.begin(), invalidSet.end(), child) == invalidSet.end()) {
                invalidSet.push_back(child);
            }
            child->valid = false;
        }
        clearBranchS(child);  // Recurse regardless of validity
    }
}

void safeRegionRrtStarDynamic::treePrune(NodePtr_dynamic newPtr)
{     
    NodePtr_dynamic ptr = newPtr;
    if( ptr->g + ptr->f > best_distance ){ // delete it and all its branches
        // std::cout<<"x:"<<ptr->coord[0]<<" y:"<<ptr->coord[1]<<" z:"<<ptr->coord[2]<<" g:"<<ptr->g<<" f:"<<ptr->f<<" bd:"<<best_distance<<std::endl;
        ptr->invalid_in_prune = true;
        ptr->valid = false;
        invalidSet.push_back(ptr);
        clearBranchS(ptr);
    }    
}

// void safeRegionRrtStarDynamic::removeInvalid4d() {
//     // Track which nodes we've processed to prevent double-deletion
//     std::unordered_set<NodePtr_dynamic> processed_nodes;
//     std::cout<<"[remove invalid debug] debug 1"<<std::endl;
//     // Create updated lists for valid nodes
//     vector<NodePtr_dynamic> UpdateNodeList;
//     vector<NodePtr_dynamic> UpdateEndList;
    
//     // Clear and rebuild the KD-tree safely
//     if (kdTree_) {
//         kd_clear(kdTree_);
//     }
//     kdTree_ = kd_create(4);  // Recreate fresh tree
//     std::cout<<"[remove invalid debug] debug 2"<<std::endl;
//     // Phase 1: Process valid nodes
//     for (auto nodeptr : NodeList) {
//         if (!nodeptr) continue;  // Skip null pointers
        
//         if (nodeptr->valid) {
//             // Insert into KD-tree
//             float pos[4] = {
//                 static_cast<float>(nodeptr->coord(0)),
//                 static_cast<float>(nodeptr->coord(1)),
//                 static_cast<float>(nodeptr->coord(2)),
//                 static_cast<float>(nodeptr->coord(3))
//             };
//             kd_insert4f(kdTree_, pos[0], pos[1], pos[2], pos[3], nodeptr);
            
//             UpdateNodeList.push_back(nodeptr);
//             if (checkEnd(nodeptr)) {
//                 UpdateEndList.push_back(nodeptr);
//             }
//         } else {
//             // Mark invalid nodes for processing
//             processed_nodes.insert(nodeptr);
//         }
//     }

//     // Phase 2: Clean up invalid nodes' relationships
//     for (auto nodeptr : invalidSet) {
//         if (!nodeptr || processed_nodes.count(nodeptr)) continue;
        
//         // Remove from parent's children list
//         if (nodeptr->preNode_ptr) {
//             auto& children = nodeptr->preNode_ptr->nxtNode_ptr;
//             children.erase(
//                 std::remove(children.begin(), children.end(), nodeptr),
//                 children.end()
//             );
//         }

//         // Clear parent pointers of children
//         for (auto childptr : nodeptr->nxtNode_ptr) {
//             if (childptr && childptr->valid && childptr->preNode_ptr == nodeptr) {
//                 childptr->preNode_ptr = nullptr;
//             }
//         }
        
//         processed_nodes.insert(nodeptr);
//     }

//     // Phase 3: Safe deletion
//     std::vector<NodePtr_dynamic> nodes_to_delete;
//     for (auto ptr : invalidSet) {
//         if (ptr && processed_nodes.count(ptr)) {
//             nodes_to_delete.push_back(ptr);
//         }
//     }
    
//     // Delete nodes after clearing all references
//     for (auto ptr : nodes_to_delete) {
//         // Clear children pointers first
//         ptr->nxtNode_ptr.clear();
//         delete ptr;
//     }

//     // Update main lists
//     NodeList = UpdateNodeList;
//     EndList = UpdateEndList;
//     invalidSet.clear();
// }

void safeRegionRrtStarDynamic::removeInvalid4d() {
    // Create updated lists for valid nodes
    vector<NodePtr_dynamic> UpdateNodeList;
    vector<NodePtr_dynamic> UpdateEndList;
    // std::cout<<"[remove invalid] debug 0"<<std::endl;
    // Clear and rebuild the KD-tree with 4D positions
    kd_clear(kdTree_);
    // std::cout<<"[remove invalid] debug 1"<<std::endl;
    // Process all nodes, keeping only valid ones
    for (auto nodeptr : NodeList) {
        if (nodeptr->valid) {
            // Use 4D coordinates for KD-tree insertion
            float pos[4] = {
                (float)nodeptr->coord(0), 
                (float)nodeptr->coord(1), 
                (float)nodeptr->coord(2),
                (float)nodeptr->coord(3)  // Time dimension
            };
            kd_insert4f(kdTree_, pos[0], pos[1], pos[2], pos[3], nodeptr);
            UpdateNodeList.push_back(nodeptr);

            if (checkEnd(nodeptr)) {
                UpdateEndList.push_back(nodeptr);
            }
        }
    }
    // std::cout<<"[remove invalid] debug 2"<<std::endl;
    // Update the main node lists
    NodeList = UpdateNodeList;
    EndList = UpdateEndList;
    // std::cout<<"[remove invalid] debug 3"<<std::endl;
    // Process invalid nodes - clean up connections
    for (auto nodeptr : invalidSet) {
        // Remove from parent's children list
        if (nodeptr->preNode_ptr != NULL) {
            auto& children = nodeptr->preNode_ptr->nxtNode_ptr;
            children.erase(
                remove(children.begin(), children.end(), nodeptr),
                children.end()
            );
        }

        // Clear parent pointers of children
        for (auto childptr : nodeptr->nxtNode_ptr) {
            if (childptr->valid && childptr->preNode_ptr == nodeptr) {
                childptr->preNode_ptr = NULL;
            }
        }
    }
    // std::cout<<"[remove invalid] debug 4"<<std::endl;
    // Delete invalid nodes only if they are not present in NodeList anymore
    std::unordered_set<NodePtr_dynamic> deleted;
    for (auto ptr : invalidSet) {
        if (ptr != NULL && deleted.find(ptr) == deleted.end()) {
            // Remove from NodeList to avoid double deletion
            NodeList.erase(std::remove(NodeList.begin(), NodeList.end(), ptr), NodeList.end());
            delete ptr;
            deleted.insert(ptr);
        }
    }
    // std::cout<<"[remove invalid] debug 5"<<std::endl;
    invalidSet.clear();
}

Eigen::Vector4d safeRegionRrtStarDynamic::getRootCoords()
{
    return root_node->coord;
}

void safeRegionRrtStarDynamic::resetRoot(Vector4d & target_coord)
{   
    // std::cout<<"[root debug] seg check 1"<<std::endl;
    std::cout<<"[root debug] prev root: "<<root_node->coord.transpose()<<std::endl;
    std::cout<<"[root debug] new root: "<<target_coord.transpose()<<std::endl;

    NodePtr_dynamic lstNode = PathList.front();
    
    if(getDis4d(lstNode, target_coord) < lstNode->radius){
        global_navi_status = true;
        return;
    }
    std::cout<<"[root debug] seg check 2"<<std::endl;

    double cost_reduction = 0;

    commit_root = target_coord;
    vector<NodePtr_dynamic> cutList;
    
    for(auto nodeptr:NodeList)
    {
        nodeptr->best = false;
    }
    std::cout<<"[root debug] seg check 3"<<std::endl;

    bool delete_root = false;

    NodePtr_dynamic temp_root = NULL;
    double min_dist = 1000000000.0;
    int root_index;
    for(int i=0; i<PathList.size(); i++)
    {
        if(getDis4d(PathList[i], target_coord) < min_dist)
        {
            min_dist = getDis4d(PathList[i], target_coord);
            temp_root = PathList[i];
            root_index = i;
        }
    }
    if (root_node == temp_root) 
    {
        std::cout << "[Debug] New root is the same as the previous root. No change needed." << std::endl;
        return; // Avoid unnecessary reassignments
    }
    if(temp_root != NULL && root_index < PathList.size())
    {
        std::cout<<"[segmentation error debug] commit target: "<<target_coord.transpose()<<std::endl;
        PathList[root_index]->best = true;
        PathList[root_index]->preNode_ptr = NULL;
        cost_reduction = PathList[root_index]->g;
        root_node = PathList[root_index];
        std::cout<<"[segmentation error debug] new root node: "<<root_node->coord.transpose()<<std::endl;

    }
        std::cout<<"[segmentation error debug] d1"<<std::endl;

    for(int i=root_index+1; i<PathList.size(); i++)
    {
        PathList[i]->best = false;
        PathList[i]->valid = false;
        cutList.push_back(PathList[i]);
    }

    std::cout<<"[root debug] seg check 4"<<std::endl;

    solutionUpdate(cost_reduction, target_coord);
    std::cout<<"[root debug] seg check 5"<<std::endl;

    for(auto nodeptr:cutList){
        invalidSet.push_back(nodeptr);
        clearBranchW(nodeptr);
    }
    std::cout<<"[root debug] seg check 6"<<std::endl;

    removeInvalid4d();
}

void safeRegionRrtStarDynamic::solutionUpdate(double cost_reduction, Vector4d target)
{
    Vector3d target3d;
    target3d[0] = target[0];
    target3d[1] = target[1];
    target3d[2] = target[2];

    for(auto nodeptr: NodeList){
        nodeptr->g -= cost_reduction;
    }
    min_distance = getDis(target3d, end_pt.head<3>());
    
    /* ---------- update ellipsoid transformation ---------- */
    /* 3D ellipsoid transformation */
    translation_inf = (target3d + end_pt.head<3>()) / 2.0;

    Eigen::Vector3d xtf, ytf, ztf, downward(0, 0, -1);
    xtf = (target3d - translation_inf).normalized();
    ytf = xtf.cross(downward).normalized();
    ztf = xtf.cross(ytf);

    rotation_inf.col(0) = xtf;
    rotation_inf.col(1) = ytf;
    rotation_inf.col(2) = ztf;

    best_distance -= cost_reduction;
}

void safeRegionRrtStarDynamic::updateHeuristicRegion(NodePtr_dynamic update_end_node)
{   
    /*  This function update the heuristic hype-ellipsoid sampling region once and better path has been found  */
    // Update the up-to-date traversal and conjugate diameter of the ellipsoid.
    // If there is no improvement in the path, maintain the heuristic unchange.

    double update_cost = update_end_node->g + getDis(update_end_node->coord.head<3>(), end_pt.head<3>()) + getDis(root_node->coord.head<3>(), commit_root.head<3>());

    if(update_cost < best_distance){  
        best_distance = update_cost;
        elli_l = best_distance;
        elli_s = sqrt(best_distance * best_distance - min_distance * min_distance);

        // release the old best path, free the best status
        if(inform_status){
            for(auto ptr:NodeList)
                ptr->best = false;
        }

        // update the nodes in the new best path to be marked as best
        NodePtr_dynamic ptr = update_end_node;
        while( ptr != NULL  ) 
        {  
            ptr->best = true;
            ptr = ptr->preNode_ptr;
        }

        best_end_ptr = update_end_node;
    }    
}

inline Vector3d safeRegionRrtStarDynamic::genSample()
{     
    Vector3d pt;
    double bias = rand_bias(eng);
    if( bias <= goal_ratio ){
        pt = end_pt.head<3>();
        return pt;
    }
    int index = rand_idx(eng);
    /*  Generate samples in a heuristic hype-ellipsoid region  */
    //    1. generate samples according to (rho, phi), which is a unit circle
    //    2. scale the unit circle to a ellipsoid
    //    3. rotate the ellipsoid
    if(!inform_status) { 
        if( bias > goal_ratio && bias <= (goal_ratio + inlier_ratio) ){ 
            // sample inside the local map's boundary
            pt(0)    = rand_x_in(eng);
            pt(1)    = rand_y_in(eng);
            pt(2)    = rand_z_in(eng);  
        }
        else{           
            // uniformly sample in all region
            pt(0)    = rand_x(eng);
            pt(1)    = rand_y(eng);
            pt(2)    = rand_z(eng);  
        }
    }
    else{ 
        /* ---------- uniform sample in 3D ellipsoid ---------- */
        double us = rand_u(eng);
        double vs = rand_v(eng);
        double phis = rand_phi(eng);

        /* inverse CDF */
        double as = elli_l / 2.0 * cbrt(us);
        double bs = elli_s / 2.0 * cbrt(us);
        double thetas = acos(1 - 2 * vs);

        pt(0) = as * sin(thetas) * cos(phis);
        pt(1) = bs * sin(thetas) * sin(phis);
        pt(2) = bs * cos(thetas);

        pt = rotation_inf * pt + translation_inf;

        pt(0) = min( max( pt(0), x_l ), x_h );
        pt(1) = min( max( pt(1), y_l ), y_h );
        pt(2) = min( max( pt(2), z_l ), z_h );
    }

    return pt;
}

inline Vector4d safeRegionRrtStarDynamic::genSample4d()
{     
    Vector3d pt;
    Vector4d pt4d;
    double bias = rand_bias(eng);

    if( bias <= goal_ratio ){
        
        pt4d.head<3>() = end_pt.head<3>();
        int index = rand_idx(eng);
        pt4d(3) = end_pt(3);
        return pt4d;
    }
    int index = rand_idx(eng);
    pt4d(3) = time_samples[index];

    /*  Generate samples in a heuristic hype-ellipsoid region  */
    //    1. generate samples according to (rho, phi), which is a unit circle
    //    2. scale the unit circle to a ellipsoid
    //    3. rotate the ellipsoid
    if(!inform_status) { 
        if( bias > goal_ratio && bias <= (goal_ratio + inlier_ratio) ){ 
            // sample inside the local map's boundary
            pt(0)    = rand_x_in(eng);
            pt(1)    = rand_y_in(eng);
            pt(2)    = rand_z_in(eng);  
        }
        else{           
            // uniformly sample in all region
            pt(0)    = rand_x(eng);
            pt(1)    = rand_y(eng);
            pt(2)    = rand_z(eng);  
        }
        pt4d(0) = pt(0);
        pt4d(1) = pt(1);
        pt4d(2) = pt(2);
        pt4d(3) = time_samples[index];
    }
    else{ 
        /* ---------- uniform sample in 3D ellipsoid ---------- */
        double us = rand_u(eng);
        double vs = rand_v(eng);
        double phis = rand_phi(eng);

        /* inverse CDF */
        double as = elli_l / 2.0 * cbrt(us);
        double bs = elli_s / 2.0 * cbrt(us);
        double thetas = acos(1 - 2 * vs);

        pt(0) = as * sin(thetas) * cos(phis);
        pt(1) = bs * sin(thetas) * sin(phis);
        pt(2) = bs * cos(thetas);

        pt = rotation_inf * pt + translation_inf;

        pt(0) = min( max( pt(0), x_l ), x_h );
        pt(1) = min( max( pt(1), y_l ), y_h );
        pt(2) = min( max( pt(2), z_l ), z_h );
        pt4d(0) = pt(0);
        pt4d(1) = pt(1);
        pt4d(2) = pt(2);
        pt4d(3) = time_samples[index];
    }

    return pt4d;
}

inline NodePtr_dynamic safeRegionRrtStarDynamic::genNewNode( Vector4d & pt_sample, NodePtr_dynamic node_nearest_ptr )
{
    double dis       = getDis4d(node_nearest_ptr, pt_sample);
    // std::cout<<"[expansion debug] sample before: "<<pt_sample.transpose()<<std::endl;
    Vector4d center;
    if(dis > max_radius)
    {
        double steer_dis = min(node_nearest_ptr->radius / dis, max_radius / dis);
        center(0) = node_nearest_ptr->coord(0) + (pt_sample(0) - node_nearest_ptr->coord(0)) * steer_dis;
        center(1) = node_nearest_ptr->coord(1) + (pt_sample(1) - node_nearest_ptr->coord(1)) * steer_dis;
        center(2) = node_nearest_ptr->coord(2) + (pt_sample(2) - node_nearest_ptr->coord(2)) * steer_dis;
    }
    else
    {
        center(0) = pt_sample(0);
        center(1) = pt_sample(1);
        center(2) = pt_sample(2);
    }
    center(3) = pt_sample(3);
    double radius_ = radiusSearch( center );
    double h_dis_  = getDis4d(center, end_pt);

    NodePtr_dynamic node_new_ptr = new Node_dynamic( center, radius_, INFINITY, h_dis_ ); 
    // std::cout<<"[expansion debug] node after: "<<center.transpose()<<std::endl;
    // std::cout<<"[new node debug] new node radius: "<<radius_<<std::endl;
    // std::cout<<"[new node debug] new node parent coord: "<<node_nearest_ptr->coord.transpose()<<std::endl;

    return node_new_ptr;
}

// bool safeRegionRrtStarDynamic::checkTrajPtCol(Vector3d & pt)
// {     
//     if(radiusSearchCollisionPred(pt) < 0.1 ) return true;
//     else return false;
// }

inline bool safeRegionRrtStarDynamic::checkEnd( NodePtr_dynamic ptr )
{    
    double distance = getDis4d(ptr, end_pt);
    
    if(distance + 0.1 < ptr->radius)
        return true;      
   
    return false;
}

inline NodePtr_dynamic safeRegionRrtStarDynamic::findNearestVertex( Vector3d & pt )
{
    float pos[3] = {(float)pt(0), (float)pt(1), (float)pt(2)};
    kdres * nearest = kd_nearestf( kdTree_, pos );

    NodePtr_dynamic node_nearest_ptr = (NodePtr_dynamic) kd_res_item_data( nearest );
    kd_res_free(nearest);
    
    return node_nearest_ptr;
}

inline NodePtr_dynamic safeRegionRrtStarDynamic::findNearestVertex4d( Vector4d & pt )
{
    float pos[4] = {(float)pt(0), (float)pt(1), (float)pt(2), (float)pt(3)};
    kdres * nearest = kd_nearest4f( kdTree_, pos[0], pos[1], pos[2], pos[3] );

    NodePtr_dynamic node_nearest_ptr = (NodePtr_dynamic) kd_res_item_data( nearest );
    kd_res_free(nearest);
    
    return node_nearest_ptr;
}

inline int safeRegionRrtStarDynamic::checkNodeRelation( double dis, NodePtr_dynamic node_1, NodePtr_dynamic node_2 )
{
// -1 indicate two nodes are connected good
//  0 indicate two nodes are not connected
//  1 indicate node_1 contains node_2, node_2 should be deleted. 

    int status;
    if( (dis + node_2->radius) == node_1->radius)
        status = 1;
    else if( (dis + 0.1) < 0.95 * (node_1->radius + node_2->radius) ) // && (dis > node_1->radius) && (dis > node_2->radius)
        status = -1;
    else
        status = 0;
    
    return status;
}

#if 1

void safeRegionRrtStarDynamic::treeRewire(NodePtr_dynamic node_new_ptr, NodePtr_dynamic node_nearest_ptr) {
    // Validate inputs
    if (!node_new_ptr || !node_nearest_ptr) return;

    // Find nearby nodes within 2*radius, considering time constraints
    float range =  2 * node_new_ptr->radius;//max_vel*delta_t;
    float pos[4] = {(float)node_new_ptr->coord[0], 
                   (float)node_new_ptr->coord[1],
                   (float)node_new_ptr->coord[2],
                   (float)node_new_ptr->coord[3]};
    
    // Only consider nodes from same or earlier timesteps
    kdres* presults = kd_nearest_range4f(kdTree_, pos[0], pos[1], pos[2], pos[3], range);
    
    vector<NodePtr_dynamic> near_nodes;
    bool is_invalid = false;

    // Check relationships with nearby nodes
    while (!kd_res_end(presults)) {
        NodePtr_dynamic near_ptr = (NodePtr_dynamic)kd_res_item_data(presults);
        
        // Skip if from future or invalid
        if (near_ptr->coord[3] > node_new_ptr->coord[3] || !near_ptr->valid) {
            kd_res_next(presults);
            continue;
        }

        double dis = getDis4d(near_ptr, node_new_ptr);
        int relation = checkNodeRelation(dis, near_ptr, node_new_ptr);
        
        // Store temporary relationship info
        near_ptr->rel_id = relation;
        near_ptr->rel_dis = dis;

        // If completely contained by another node, invalidate
        if (relation == 1) {
            node_new_ptr->invalid_in_rewire = true;
            node_new_ptr->valid = false;
            is_invalid = true;
            break;
        }

        near_nodes.push_back(near_ptr);
        kd_res_next(presults);
    }
    kd_res_free(presults);

    if (is_invalid) {
        for (auto node : near_nodes) {
            node->rel_id = -2;
            node->rel_dis = -1.0;
        }
        return;
    }

    // Initial connection to nearest node
    double min_cost = node_nearest_ptr->g + getDis4d(node_nearest_ptr, node_new_ptr);
    node_new_ptr->preNode_ptr = node_nearest_ptr;
    node_new_ptr->g = min_cost;
    node_nearest_ptr->nxtNode_ptr.push_back(node_new_ptr);
    NodePtr_dynamic best_parent = node_nearest_ptr;

    // Find optimal parent among nearby nodes
    for (auto near_ptr : near_nodes) {
        // Skip if from future (shouldn't happen due to earlier check)
        if (near_ptr->coord[3] > node_new_ptr->coord[3]) continue;

        double dis = near_ptr->rel_dis;
        double cost = near_ptr->g + dis;

        // Check if this node provides a better path
        if (near_ptr->rel_id == -1 && cost < min_cost) {
            min_cost = cost;
            best_parent->nxtNode_ptr.pop_back();  // Remove from old parent
            best_parent = near_ptr;
            node_new_ptr->preNode_ptr = best_parent;
            node_new_ptr->g = min_cost;
            best_parent->nxtNode_ptr.push_back(node_new_ptr);
        }

        // Reset temporary values
        near_ptr->rel_id = -2;
        near_ptr->rel_dis = -1.0;
    }

    // Rewire nearby nodes through new node
    for (auto near_ptr : near_nodes) {
        if (!near_ptr->valid || near_ptr->coord[3] < node_new_ptr->coord[3]) {
            // Only rewire nodes that are from earlier timesteps
            
            double dis = getDis4d(near_ptr, node_new_ptr);
            double cost = dis + node_new_ptr->g;

            // Check if rewiring provides a better path
            if (cost < near_ptr->g && !isSuccessor(near_ptr, node_new_ptr)) {
                // Remove from old parent
                if (near_ptr->preNode_ptr) {
                    auto& children = near_ptr->preNode_ptr->nxtNode_ptr;
                    children.erase(remove(children.begin(), children.end(), near_ptr), 
                                 children.end());
                }

                // Connect to new parent
                near_ptr->preNode_ptr = node_new_ptr;
                near_ptr->g = cost;
                node_new_ptr->nxtNode_ptr.push_back(near_ptr);
            }
        }
    }
}
#endif

void safeRegionRrtStarDynamic::recordNode(NodePtr_dynamic new_node)
{
    NodeList.push_back(new_node);
}

void safeRegionRrtStarDynamic::tracePath()
{   
    vector<NodePtr_dynamic> feasibleEndList;      
    for(auto endPtr: EndList)
    {
        if( (checkValidEnd(endPtr) == false) || (checkEnd(endPtr) == false) || (endPtr->valid == false) )
        {
            std::cout<<"[trace path debug] endPtr pos"<<endPtr->coord.transpose()<<" radius: "<<endPtr->radius<<std::endl;
            std::cout<<"[trace path debug] "<<checkValidEnd(endPtr)<<" : "<<checkEnd(endPtr)<<" : "<<endPtr->valid<<std::endl;
            continue;
        }
        else
            feasibleEndList.push_back(endPtr);
    }

    if( int(feasibleEndList.size()) == 0 )
    {
        std::cout<<"[trace path] can't find a feasible path. "<<std::endl;
        path_exist_status = false;
        best_distance = INFINITY;
        inform_status = false;
        EndList.clear();
        Path   = Eigen::MatrixXd::Identity(3,3);
        Radius = Eigen::VectorXd::Zero(3);
        return;
    }

    EndList = feasibleEndList;

    best_end_ptr = feasibleEndList[0];
    double best_cost = INFINITY;
    for(auto nodeptr: feasibleEndList)
    {   
        double cost = (nodeptr->g + getDis4d(nodeptr, end_pt) + getDis4d(root_node, commit_root) );
        if( cost < best_cost )
        {
            best_end_ptr  = nodeptr;
            best_cost = cost;
            best_distance = best_cost;
        }
    }

    NodePtr_dynamic ptr = best_end_ptr;

    /*  stack the path (corridor) data for trajectory generation  */
    int idx = 0;
    PathList.clear();

    while( ptr != NULL ) {
        // std::cout<<"point: "<<ptr->coord[0]<<" : "<<ptr->coord[1]<<" : "<<ptr->coord[2]<<std::endl; 
        PathList.push_back( ptr );
        ptr = ptr->preNode_ptr;
        idx ++;
    }

    Path.resize(idx, 4) ;
    Radius.resize( idx );
    idx = 0;
    
    ptr = best_end_ptr;
    idx ++ ;

    while( ptr != NULL ){      
          Path.row(Path.rows() - idx) = ptr->coord;
          Radius(Radius.size() - idx) = ptr->radius;
          
          ptr = ptr->preNode_ptr;
          idx ++;
    }

    path_exist_status = true;
}

void safeRegionRrtStarDynamic::treeDestruct()
{
    if(kdTree_ != NULL)
    {
        kd_free(kdTree_);
    }
    
    for( int i = 0; i < int(NodeList.size()); i++)
    {   
        NodePtr_dynamic ptr = NodeList[i];
        if(ptr != NULL)
        {
            delete ptr;
        }
    }
}

inline double safeRegionRrtStarDynamic::checkRadius(Vector4d & pt)
{
    return radiusSearch( pt );
}

inline int safeRegionRrtStarDynamic::checkNodeUpdate(double new_radius, double old_radius)
{
    if( new_radius < safety_margin)  
        return -1; // for not qualified, this node should be deleted
    else if( new_radius < old_radius )
        return 0;  // for node radius shrink, this node should be inspected for connection with others 
    else 
        return 1;  // for no difference, nothing should be changed
}


inline bool safeRegionRrtStarDynamic::isSuccessor(NodePtr_dynamic curPtr, NodePtr_dynamic nearPtr) // check if curPtr is father of nearPtr
{     
    NodePtr_dynamic prePtr = nearPtr->preNode_ptr;
    
    while(prePtr != NULL)
    {
        if(prePtr == curPtr) return true;
        else
          prePtr = prePtr->preNode_ptr;
    }

    return false;
}

inline bool safeRegionRrtStarDynamic::checkValidEnd(NodePtr_dynamic endPtr)
{
    NodePtr_dynamic ptr = endPtr;

    while( ptr != NULL )
    {   
        if(!ptr->valid) 
        {
            std::cout<<"ptr invalid: "<<ptr->coord.transpose()<<std::endl;
            return false;
        }
        
        if( getDis4d( ptr, root_node->coord ) < ptr->radius )
        {
            return true;
        }

        ptr = ptr->preNode_ptr;
    }

    return false;  
}

void safeRegionRrtStarDynamic::SafeRegionExpansion( double time_limit )
{   
    /*  The initialization of the safe region RRT* tree  */
    auto time_bef_expand = std::chrono::steady_clock::now();
    kdTree_ = kd_create(4);
    commit_root.head<3>()  = start_pt;
    double time_start = time_samples[0];
    Vector4d start_pt_4d;
    start_pt_4d(0) = start_pt(0);
    start_pt_4d(1) = start_pt(1);
    start_pt_4d(2) = start_pt(2);
    start_pt_4d(3) = time_samples[0];
    root_node = new Node_dynamic(start_pt_4d, radiusSearch( start_pt_4d ), 0.0, min_distance); // TODO: radiusSearch

    recordNode(root_node);

    float pos4d[4] = {(float)root_node->coord(0), (float)root_node->coord(1), (float)root_node->coord(2), (float)time_start};
    kd_insertf(kdTree_, pos4d, root_node);
    int iter_count;

    for( iter_count = 0; iter_count < max_samples; iter_count ++)
    {     
        auto time_in_expand = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_in_expand - time_bef_expand).count()*0.001;
        // if( elapsed > time_limit ) 
        // {
        //     std::cout<<"[expansion debug] time elapsed: "<<std::endl;
        //     break;
        // }
        
        Vector4d pt_sample = genSample4d();
        // sampleList.push_back(pt_sample);
        
        NodePtr_dynamic node_nearest_ptr = findNearestVertex4d(pt_sample); // TODO: use 4D kd-tree to find the nearest vertex
        
        if(!node_nearest_ptr->valid || node_nearest_ptr == NULL || node_nearest_ptr->coord(3) > pt_sample(3))
        {
            // std::cout<<"[expansion debug] condition 1: "<<node_nearest_ptr->coord(3)<<" : "<<pt_sample(3)<<std::endl;
            continue;
        }

        NodePtr_dynamic node_new_ptr = genNewNode(pt_sample, node_nearest_ptr); // TODO: use 4D node to generate new node
        if( node_new_ptr->coord(2) < (z_l + safety_margin) || node_new_ptr->radius < safety_margin )
        {
            bool a = node_new_ptr->coord(2) < z_l;
            bool b = node_new_ptr->radius < safety_margin;
            // std::cout<<"[expansion debug] condition 2: "<<a<<" : "<<b<<std::endl;
            // std::cout<<"[expansion debug] condition 2 coords: "<<node_new_ptr->coord.transpose()<<std::endl;
            // std::cout<<"[expansion debug] condition 2 radius: "<<node_new_ptr->radius<<std::endl;
            // std::cout<<"[expansion debug] condition 2 z_l: "<<z_l<<std::endl;

            continue;
        }
        
        treeRewire(node_new_ptr, node_nearest_ptr);  
        
        if( ! node_new_ptr->valid)
        {
            continue;
        }

        if(checkEnd( node_new_ptr ))
        {
            if( !inform_status ) best_end_ptr = node_new_ptr;
            
            EndList.push_back(node_new_ptr);
            updateHeuristicRegion(node_new_ptr);
            inform_status = true;  
        }
        
        pos4d[0] = (float)node_new_ptr->coord(0);
        pos4d[1] = (float)node_new_ptr->coord(1);
        pos4d[2] = (float)node_new_ptr->coord(2);
        pos4d[3] = (float)node_new_ptr->coord(3); // time dimension
        kd_insertf(kdTree_, pos4d, node_new_ptr);

        recordNode(node_new_ptr);
        std::cout<<"inserted node coord: "<<node_new_ptr->coord.transpose()<<std::endl;
        std::cout<<"inserted node parent: "<<node_new_ptr->preNode_ptr->coord.transpose()<<std::endl;

        selectedNodeList.push_back(node_new_ptr->coord);
        treePrune(node_new_ptr);      

        if( int(invalidSet.size()) >= cach_size) removeInvalid4d();
    }

    removeInvalid4d();
    tracePath();

}

void safeRegionRrtStarDynamic::SafeRegionRefine( double time_limit )
{   
    /*  Every time the refine function is called, new samples are continuously generated and the tree is continuously rewired, hope to get a better solution  */
    /*  The refine process is mainly same as the initialization of the tree  */
    auto time_bef_refine = std::chrono::steady_clock::now();
    std::cout<<"in refine loop"<<std::endl;

    float pos[4];
    while( true )
    {     
        std::cout<<"[refine seg debug 1]"<<std::endl;
        auto time_in_refine = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_in_refine - time_bef_refine).count()*0.001;
        if( elapsed > time_limit )
        {
            // std::cout<<"[refine debug] time in refine loop: "<<elapsed<<" time limit: "<<time_limit<<std::endl;
            break;
        } 
        std::cout<<"[refine seg debug 1]"<<std::endl;
        Vector4d pt_sample =  genSample4d();
        NodePtr_dynamic node_nearest_ptr = findNearestVertex4d(pt_sample);
        std::cout<<"[refine seg debug 11]"<<std::endl;
        if(!node_nearest_ptr->valid || node_nearest_ptr == NULL || node_nearest_ptr->coord(3) > pt_sample(3)) continue;
        std::cout<<"[refine seg debug 12]"<<std::endl;          
        NodePtr_dynamic node_new_ptr  =  genNewNode(pt_sample, node_nearest_ptr); 
        std::cout<<"[refine seg debug 13]"<<std::endl;
        if( node_new_ptr->coord(2) < z_l  || node_new_ptr->radius < safety_margin ) continue;
        std::cout<<"[refine seg debug 14]"<<std::endl;  
        treeRewire(node_new_ptr, node_nearest_ptr);  
        std::cout<<"[refine seg debug 15]"<<std::endl;
        if( node_new_ptr->valid == false ) continue;
        std::cout<<"[refine seg debug 16]"<<std::endl;
        if(checkEnd( node_new_ptr )){
            if( !inform_status ) 
                best_end_ptr = node_new_ptr;
            
            EndList.push_back(node_new_ptr);
            updateHeuristicRegion(node_new_ptr);
            inform_status = true;  
        }
        std::cout<<"[refine seg debug 17]"<<std::endl;
        pos[0] = (float)node_new_ptr->coord(0);
        pos[1] = (float)node_new_ptr->coord(1);
        pos[2] = (float)node_new_ptr->coord(2);
        pos[3] = (float)node_new_ptr->coord(3);
        kd_insert4f(kdTree_, pos[0], pos[1], pos[2], pos[3], node_new_ptr);
        std::cout<<"[refine seg debug 18]"<<std::endl;
        recordNode(node_new_ptr);
        treePrune(node_new_ptr);      
        if( int(invalidSet.size()) >= cach_size) removeInvalid4d();
        std::cout<<"[refine seg debug 19]"<<std::endl;
  }
    std::cout<<"[refine seg debug 2]"<<std::endl;
    removeInvalid4d();  
    std::cout<<"[refine seg debug 3]"<<std::endl;  
    tracePath();

}

void safeRegionRrtStarDynamic::SafeRegionEvaluate(double time_limit) {
    if (!path_exist_status) return;

    auto time_bef_evaluate = chrono::steady_clock::now();
    std::vector<pair<Vector4d, double>> fail_node_list;  // Now stores 4D coordinates

    while (true) 
    {
        // Check all nodes in current best path
        std::cout<<"[evaluate debug] debug 0"<<std::endl;
        for (auto ptr : PathList) {
            NodePtr_dynamic pre_ptr = ptr->preNode_ptr;
            if (!pre_ptr) continue;  // Skip root node

            // Verify temporal ordering (parent must come before child in time)
            if (pre_ptr->coord[3] > ptr->coord[3]) {
                std::cout<<"[evaluate debug] debug in temporal ordering loop"<<std::endl;
                ptr->valid = false;
                invalidSet.push_back(ptr);
                clearBranchS(ptr);
                fail_node_list.emplace_back(ptr->coord, ptr->radius);
                continue;
            }
            std::cout<<"[evaluate debug] debug 0a"<<std::endl;

            // Check node validity in 4D space-time
            double update_radius = radiusSearch(ptr->coord); 
            std::cout<<"[evaluate debug] debug 0b"<<std::endl;
            int ret = checkNodeUpdate(update_radius, ptr->radius);
            double old_radius = ptr->radius;
            ptr->radius = update_radius;
            std::cout<<"[evaluate debug] debug 0c"<<std::endl;
            if (ret == -1) {  // Node became invalid
                std::cout<<"[evaluate debug] debug in node invalid loop"<<std::endl;
                ptr->valid = false;
                invalidSet.push_back(ptr);
                clearBranchS(ptr);
                fail_node_list.emplace_back(ptr->coord, old_radius);
                continue;
            }
            std::cout<<"[evaluate debug] debug 0d"<<std::endl;
            // Check parent connection with 4D distance
            if (checkNodeRelation(getDis4d(ptr, pre_ptr), ptr, pre_ptr) != -1) {
                std::cout<<"[evaluate debug] debug in node relations loop"<<std::endl;
                if (ptr->valid) {
                    std::cout<<"[evaluate debug] debug in node relations loop 2"<<std::endl;
                    ptr->valid = false;
                    invalidSet.push_back(ptr);
                    clearBranchS(ptr);
                    fail_node_list.emplace_back(ptr->coord, old_radius);
                }
                continue;
            }
            std::cout<<"[evaluate debug] debug 0e"<<std::endl;
            // Check child connections
            for (auto childptr : ptr->nxtNode_ptr) {
                // Verify temporal ordering (child must come after parent)
                if (childptr->coord[3] < ptr->coord[3]) {
                    std::cout<<"[evaluate debug] debug checking childptr loop"<<std::endl;
                    childptr->valid = false;
                    invalidSet.push_back(childptr);
                    clearBranchS(childptr);
                    fail_node_list.emplace_back(childptr->coord, childptr->radius);
                    continue;
                }

                if (checkNodeRelation(getDis4d(ptr, childptr), ptr, childptr) != -1) {
                    if (childptr->valid) {
                        childptr->valid = false;
                        invalidSet.push_back(childptr);
                        clearBranchS(childptr);
                        fail_node_list.emplace_back(childptr->coord, childptr->radius);
                    }
                }
            }
            std::cout<<"[evaluate debug] debug 0f"<<std::endl;
        }
        std::cout<<"[evaluate debug] debug 1"<<std::endl;
        // Check if current path is still fully valid
        bool path_valid = all_of(PathList.begin(), PathList.end(), 
                               [](const NodePtr_dynamic& ptr) { return ptr->valid; });

        std::cout<<"[evaluate debug] debug 2"<<std::endl;

        // Timeout check
        auto current_time = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(current_time - time_bef_evaluate).count();
        if (elapsed > time_limit) {
            path_exist_status = false;
            inform_status = false;
            best_distance = INFINITY;
            break;
        }
        std::cout<<"[evaluate debug] debug 3"<<std::endl;

        if (path_valid) break;
        std::cout<<"[evaluate debug] debug 4"<<std::endl;

        // Find new feasible ends considering time constraints
        vector<NodePtr_dynamic> feasibleEndList;
        for (auto endPtr : EndList) {
            if (endPtr->valid && checkEnd(endPtr) && 
                endPtr->coord[3] <= time_samples.back()) {  // Check time bounds
                feasibleEndList.push_back(endPtr);
            }
        }
        std::cout<<"[evaluate debug] debug 5"<<std::endl;

        if (feasibleEndList.empty()) {
            path_exist_status = false;
            inform_status = false;
            best_distance = INFINITY;
            break;
        }
        std::cout<<"[evaluate debug] debug 6"<<std::endl;

        // Select best path considering 4D cost (distance + time)
        best_end_ptr = *min_element(feasibleEndList.begin(), feasibleEndList.end(),
            [this](const NodePtr_dynamic& a, const NodePtr_dynamic& b) {
                return (a->g + getDis4d(a, end_pt)) < (b->g + getDis4d(b, end_pt));
            });
        std::cout<<"[evaluate debug] debug 7"<<std::endl;

        best_distance = best_end_ptr->g + getDis4d(best_end_ptr, end_pt);
        std::cout<<"[evaluate debug] debug 8"<<std::endl;

        // Rebuild PathList with temporal ordering
        PathList.clear();
        for (NodePtr_dynamic ptr = best_end_ptr; ptr != nullptr; ptr = ptr->preNode_ptr) {
            PathList.push_back(ptr);
        }
        reverse(PathList.begin(), PathList.end());  // Ensure chronological order
        std::cout<<"[evaluate debug] debug 9"<<std::endl;

    }

    // Calculate remaining time for repair
    auto time_aft_evaluate = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(time_aft_evaluate - time_bef_evaluate).count();
    double repair_limit = max(0.0, time_limit - elapsed);

    // Clean up and repair
    removeInvalid4d();
    treeRepair(repair_limit, fail_node_list);
    tracePath();
}

void safeRegionRrtStarDynamic::treeRepair(double time_limit, vector<pair<Vector4d, double>>& node_list) {
    /* Repair the tree around failed nodes in 4D space-time */
    auto time_bef_repair = chrono::steady_clock::now();
    
    for (const auto& [center, radius] : node_list) {
        // Check time limit
        auto time_now = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(time_now - time_bef_repair).count();
        if (elapsed > time_limit) break;

        // 4D range search around failed node
        float pos[4] = {(float)center[0], (float)center[1], 
                       (float)center[2], (float)center[3]};
        float range =  radius * 2.0f; // 2*max_vel * delta_t;
        
        kdres* presults = kd_nearest_range4f(kdTree_, pos[0], pos[1], pos[2], pos[3], range);

        while (!kd_res_end(presults)) {
            NodePtr_dynamic ptr = (NodePtr_dynamic)kd_res_item_data(presults);
            kd_res_next(presults);

            // Skip invalid or root nodes
            if (!ptr->valid || ptr == root_node) continue;

            NodePtr_dynamic pre_ptr = ptr->preNode_ptr;
            if (pre_ptr == root_node) continue;

            // Verify temporal ordering
            if (pre_ptr && pre_ptr->coord[3] > ptr->coord[3]) {
                ptr->valid = false;
                invalidSet.push_back(ptr);
                clearBranchS(ptr);
                continue;
            }

            // 4D collision checking
            double update_radius = radiusSearch(ptr->coord);
            std::cout<<"[repair debug] after radius search"<<std::endl;
            int ret = checkNodeUpdate(update_radius, ptr->radius);
            ptr->radius = update_radius;
            std::cout<<"[repair debug] debug 1"<<std::endl;

            if (ret == -1) { // Node became invalid
                ptr->valid = false;
                invalidSet.push_back(ptr);
                clearBranchS(ptr);
                continue;
            }
            std::cout<<"[repair debug] debug 2"<<std::endl;
            // Check parent connection with 4D distance
            if (pre_ptr && checkNodeRelation(getDis4d(pre_ptr, ptr), pre_ptr, ptr) != -1) {
                ptr->valid = false;
                invalidSet.push_back(ptr);
                clearBranchS(ptr);
                continue;
            }
            std::cout<<"[repair debug] debug 3"<<std::endl;
            // Check child connections
            for (auto childptr : ptr->nxtNode_ptr) {
                // Verify temporal ordering
                if (childptr->coord[3] < ptr->coord[3]) {
                    childptr->valid = false;
                    invalidSet.push_back(childptr);
                    clearBranchS(childptr);
                    continue;
                }
                std::cout<<"[repair debug] debug 3a"<<std::endl;
                if (checkNodeRelation(getDis4d(ptr, childptr), ptr, childptr) != -1) {
                    childptr->valid = false;
                    invalidSet.push_back(childptr);
                    clearBranchS(childptr);
                }
                std::cout<<"[repair debug] debug 3b"<<std::endl;
            }
        }
        std::cout<<"[repair debug] debug 4"<<std::endl;
        kd_res_free(presults);
        std::cout<<"[repair debug] debug 5"<<std::endl;
    }
    std::cout<<"[repair debug] debug 6"<<std::endl;

    removeInvalid4d();
    std::cout<<"[repair debug] debug 7"<<std::endl;

}