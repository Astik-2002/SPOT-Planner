#include "rrt_path_finder/corridor_finder_dynamic.h"
#include <algorithm>
#include <chrono>
#include <memory>
using namespace Eigen;
using namespace std;

safeRegionRrtStarDynamic::safeRegionRrtStarDynamic() { 
    cach_size = 100;
    kdTree_ = nullptr;
}

safeRegionRrtStarDynamic::~safeRegionRrtStarDynamic() { }

Eigen::Vector3d safeRegionRrtStarDynamic::computeNoiseStd(const Eigen::Vector3d& point) {
    double x = point.x();
    double sigma_x = 0.001063 + 0.0007278 * x + 0.003949 * x * x;
    double sigma_y = 0.04;
    double sigma_z = 0.04;
    return Eigen::Vector3d(sigma_x, sigma_y, sigma_z);
}

void safeRegionRrtStarDynamic::setParam(double safety_margin_, double search_margin_, double deltat, double sample_range_, double h_fov_, double v_fov_, bool uncertanity_) {   
    safety_margin = safety_margin_;
    search_margin = search_margin_;
    sample_range = sample_range_;
    h_fov = h_fov_;
    v_fov = v_fov_;
    uncertanity = uncertanity_;
    best_distance = INFINITY;
    delta_t = deltat;
}

void safeRegionRrtStarDynamic::reset() 
{
    treeDestruct();    
    NodeList.clear();
    EndList.clear();
    invalidSet.clear();
    PathList.clear();
    best_end_ptr = make_shared<Node_dynamic>();
    root_node = make_shared<Node_dynamic>();
    path_exist_status = true;
    inform_status = false;
    global_navi_status = false;
    best_distance = INFINITY;  
}

void safeRegionRrtStarDynamic::setStartPt(Vector3d startPt, Vector3d endPt) {
    start_pt = startPt;
    end_pt.head<3>() = endPt; 
    double d = getDis(start_pt, end_pt.head<3>());
    rand_x_in = uniform_real_distribution<double>(start_pt(0) - sample_range, start_pt(0) + sample_range);
    rand_y_in = uniform_real_distribution<double>(start_pt(1) - sample_range, start_pt(1) + sample_range);
}

void safeRegionRrtStarDynamic::setPt(Vector3d startPt, Vector3d endPt, double xl, double xh, double yl, double yh, double zl, double zh,
                       double local_range, int max_iter, double sample_portion, double goal_portion, double yaw, double _average_vel, double _max_vel, double weight_t) 
{
    start_pt = startPt;
    end_pt   = endPt; 
    // std::cout<<"start pt: "<<startPt.transpose()<<" : endpt: "<<endPt.transpose()<<std::endl;

    x_l = xl; x_h = xh;
    y_l = yl; y_h = yh;
    z_l = zl; z_h = zh;

    bias_l = 0.0; bias_h = 1.0;
    
    eng = default_random_engine(rd());
    rand_x = uniform_real_distribution<double>(x_l, x_h);
    rand_y = uniform_real_distribution<double>(y_l, y_h);
    rand_z = rand_z_in = uniform_real_distribution<double>(z_l, z_h);
    rand_bias = uniform_real_distribution<double>(bias_l, bias_h);

    rand_x_in = uniform_real_distribution<double>(start_pt(0) - sample_range, start_pt(0) + sample_range);
    rand_y_in = uniform_real_distribution<double>(start_pt(1) - sample_range, start_pt(1) + sample_range);
    average_vel = _average_vel;
    weightT = weight_t; // cannot be more than 1
    assert(weightT > 0.0 && weightT < 1.0);
    max_vel = _max_vel;
    max_radius = 2*max_vel*delta_t;
    min_distance = (end_pt - start_pt).norm() * ((1-weightT) + weightT / max_vel);
    translation_inf = (start_pt + end_pt) / 2.0;

    Eigen::Vector3d xtf, ytf, ztf, downward(0, 0, -1);
    xtf = (end_pt - translation_inf).normalized();
    ytf = xtf.cross(downward).normalized();
    ztf = xtf.cross(ytf);

    rotation_inf.col(0) = xtf;
    rotation_inf.col(1) = ytf;
    rotation_inf.col(2) = ztf;

    sample_range = local_range;
    max_samples = max_iter;
    inlier_ratio = sample_portion;
    goal_ratio = goal_portion;
    current_yaw = yaw;
    // std::cout << "[setPt] start: " << start_pt.transpose()
    //       << ", end: " << end_pt.transpose()
    //       << ", min_distance (raw): " << (end_pt - start_pt).norm()
    //       << ", time_weight: " << weightT
    //       << ", min_distance (final): " << min_distance
    //       << std::endl;
}

inline double safeRegionRrtStarDynamic::getDis(const NodePtr_dynamic node1, const NodePtr_dynamic node2) {
    return sqrt(pow(node1->coord(0) - node2->coord(0), 2) + pow(node1->coord(1) - node2->coord(1), 2) + pow(node1->coord(2) - node2->coord(2), 2));
}

inline double safeRegionRrtStarDynamic::getDis(const NodePtr_dynamic node1, const Vector3d & pt) {
    return sqrt(pow(node1->coord(0) - pt(0), 2) + pow(node1->coord(1) - pt(1), 2) + pow(node1->coord(2) - pt(2), 2));
}

inline double safeRegionRrtStarDynamic::getDis(const Vector3d & p1, const Vector3d & p2) {
    return sqrt(pow(p1(0) - p2(0), 2) + pow(p1(1) - p2(1), 2) + pow(p1(2) - p2(2), 2));
}

inline double safeRegionRrtStarDynamic::getDis4d(const NodePtr_dynamic node1, const NodePtr_dynamic node2) {
    return sqrt(pow(node1->coord(0) - node2->coord(0), 2) + pow(node1->coord(1) - node2->coord(1), 2) + pow(node1->coord(2) - node2->coord(2), 2) + pow(node1->coord(3) - node2->coord(3), 2));
}

inline double safeRegionRrtStarDynamic::getDis4d(const NodePtr_dynamic node1, const Vector4d & pt) {
    return sqrt(pow(node1->coord(0) - pt(0), 2) + pow(node1->coord(1) - pt(1), 2) + pow(node1->coord(2) - pt(2), 2) + pow(node1->coord(3) - pt(3), 2));
}

inline double safeRegionRrtStarDynamic::getDis4d(const Vector4d & p1, const Vector4d & p2) {
    return sqrt(pow(p1(0) - p2(0), 2) + pow(p1(1) - p2(1), 2) + pow(p1(2) - p2(2), 2) + pow(p1(3) - p2(3), 2));
}

inline bool safeRegionRrtStarDynamic::isSuccessor(NodePtr_dynamic curPtr, NodePtr_dynamic nearPtr) {
    NodePtr_dynamic prePtr = nearPtr->preNode_ptr;
    
    while(prePtr) {
        if(prePtr == curPtr) return true;
        prePtr = prePtr->preNode_ptr;
    }
    return false;
}

inline bool safeRegionRrtStarDynamic::checkValidEnd(NodePtr_dynamic endPtr) {
    NodePtr_dynamic ptr = endPtr;
    while(ptr) {   
        if(!ptr->valid) 
            return false;
        
        auto st_pair = getDisMetric(root_node, ptr);
        if(ptr == root_node)
            return true;
        if((ptr->coord[3] - root_node->coord[3]) <= 2*delta_t && st_pair.first <= max_vel*(ptr->coord[3] - root_node->coord[3]))
            return true;
        ptr = ptr->preNode_ptr;
    }
    return false;  
}

inline double safeRegionRrtStarDynamic::checkRadius(Vector4d & pt) {
    return radiusSearch(pt);
}

inline int safeRegionRrtStarDynamic::checkNodeUpdate(double new_radius, double old_radius) {
    if(new_radius < safety_margin)  
        return -1;
    else if(new_radius < old_radius)
        return 0;
    else 
        return 1;
}

void safeRegionRrtStarDynamic::setInputStatic(pcl::PointCloud<pcl::PointXYZ> cloud_in) 
{     
    if(cloud_in.points.size() > 0)
    {
        kdtreeForMap.setInputCloud(cloud_in.makeShared());
    }  
    CloudIn = cloud_in;
}

void safeRegionRrtStarDynamic::setInputDynamic(
    const vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& _dynamic_points,
    const Eigen::Vector3d& origin,
    double start_time) 
{
    dynamic_points = _dynamic_points;
    pcd_origin = origin;
    PCDstartTime = start_time;
    buildTemporalGrid(dynamic_points);
}

void safeRegionRrtStarDynamic::buildTemporalGrid(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& dynamic_points) 
{
    temporal_grid.clear();
    double t_start = PCDstartTime;
    double t_end = t_start + 30.0;
    // if(root_node)
    // {
    //     t_end = t_start + 2*getDis(pcd_origin, end_pt)/max_vel;
    // }
    for (const auto& dyn_pair : dynamic_points) {
        const Eigen::Vector3d& pos0 = dyn_pair.first;
        const Eigen::Vector3d& vel = dyn_pair.second;

        // Discretize time and project obstacles
        for (double t = t_start; t <= t_end; t += time_resolution) {
            Eigen::Vector3d pos_t = pos0 + vel * (t - PCDstartTime);
            
            // Compute grid indices
            int t_bin = static_cast<int>(t / time_resolution);
            int x_grid = static_cast<int>(pos_t.x() / cell_size);
            int y_grid = static_cast<int>(pos_t.y() / cell_size);
            int z_grid = static_cast<int>(pos_t.z() / cell_size);
            auto grid_key = std::make_tuple(x_grid, y_grid, z_grid);

            // Store in temporal grid
            temporal_grid[t_bin][grid_key].push_back(pos_t);
        }
    }
}

double safeRegionRrtStarDynamic::queryDynamicObstacles(const Eigen::Vector4d& query_pt) 
{
    double t_q = query_pt(3);
    Eigen::Vector3d query_pos = query_pt.head<3>();
    double min_dist = INFINITY;
    for(double t = t_q; t < t_q + 2*delta_t; t += time_resolution)
    {
        int t_bin = static_cast<int>(t / time_resolution);
        if (!temporal_grid.count(t_bin)) 
        {
            auto it = temporal_grid.lower_bound(t_bin);  // Returns iterator to first key >= t_bin

            // Handle edge cases:
            if (it == temporal_grid.begin()) {
                t_bin = it->first;  // Only bin after t_bin exists
            } 
            else if (it == temporal_grid.end()) {
                t_bin = std::prev(it)->first;  // Only bin before t_bin exists
            }
            else {
                // Compare distances to previous and current bins
                int prev_bin = std::prev(it)->first;
                int next_bin = it->first;
                t_bin = (abs(t_bin - prev_bin) < abs(t_bin - next_bin)) ? prev_bin : next_bin;
            }
        }
        int x_grid = static_cast<int>(query_pos.x() / cell_size);
        int y_grid = static_cast<int>(query_pos.y() / cell_size);
        int z_grid = static_cast<int>(query_pos.z() / cell_size);

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    auto key = std::make_tuple(x_grid + dx, y_grid + dy, z_grid + dz);
                    
                    if (temporal_grid[t_bin].count(key)) {
                        for (const auto& obstacle_pos : temporal_grid[t_bin][key]) {
                            double dist = (query_pos - obstacle_pos).norm();
                            min_dist = std::min(min_dist, dist);
                        }
                    }
                }
            }
        }

    }   
    
    return min_dist;
}

inline pair<double, double> safeRegionRrtStarDynamic::getDisMetric(const NodePtr_dynamic node_p, const NodePtr_dynamic node_c) {
    Eigen::Vector3d np = node_p->coord.head<3>();
    Eigen::Vector3d nc = node_c->coord.head<3>();
    double dS = (np - nc).norm();
    double dT = node_c->coord[3] - node_p->coord[3];
    return make_pair(dS, dT);
}

inline pair<double, double> safeRegionRrtStarDynamic::getDisMetric(const Eigen::Vector4d node_p, const Eigen::Vector4d node_c) {
    Eigen::Vector3d np = node_p.head<3>();
    Eigen::Vector3d nc = node_c.head<3>();
    double dS = (np - nc).norm();
    double dT = node_c[3] - node_p[3];
    return make_pair(dS, dT);
}

inline double safeRegionRrtStarDynamic::radiusSearch(Vector4d & search_Pt, bool traj_check) {     
    double d_origin = (pcd_origin - search_Pt.head<3>()).norm();
    if(d_origin > sample_range + max_radius)
    {
        return max_radius;
    }
    pcl::PointXYZ searchPoint;
    searchPoint.x = search_Pt(0);
    searchPoint.y = search_Pt(1);
    searchPoint.z = search_Pt(2);
    double time_point = search_Pt(3);
    pointIdxRadiusSearch.clear();
    pointRadiusSquaredDistance.clear();
    kdtreeForMap.nearestKSearch(searchPoint, 1, pointIdxRadiusSearch, pointRadiusSquaredDistance);
    if(time_point < PCDstartTime) {
        return -1.00;
    }
    double map_dist = INFINITY;
    if (!pointRadiusSquaredDistance.empty()) {
        map_dist = sqrt(pointRadiusSquaredDistance[0]);
    }
    if(traj_check == false)
    {
        if(map_dist < (safety_margin + search_margin))
        {
            return 0.00;
        }
    }
    double dynamic_dist = INFINITY;
    dynamic_dist = queryDynamicObstacles(search_Pt);
    // std::cout<<"radius check: hashing check: "<<dynamic_dist<<std::endl; // " brute force check: "<<min_dynamic_dist<<std::endl;
    double rad = std::min(map_dist, dynamic_dist);
    return std::min(rad - search_margin, max_radius);
}

void safeRegionRrtStarDynamic::checkingRadius(NodePtr_dynamic node)
{
    pcl::PointXYZ searchPoint;
    searchPoint.x = node->coord(0);
    searchPoint.y = node->coord(1);
    searchPoint.z = node->coord(2);
    double time_point = node->coord(3);

    pointIdxRadiusSearch.clear();
    pointRadiusSquaredDistance.clear();
    kdtreeForMap.nearestKSearch(searchPoint, 1, pointIdxRadiusSearch, pointRadiusSquaredDistance);
    double map_dist = INFINITY;
    if (!pointRadiusSquaredDistance.empty()) {
        map_dist = sqrt(pointRadiusSquaredDistance[0]);
    }

    double min_dynamic_dist = INFINITY;
    for (const auto& dyn_obstacle_pair : dynamic_points) {
        const Eigen::Vector3d& position = dyn_obstacle_pair.first;
        const Eigen::Vector3d& velocity = dyn_obstacle_pair.second;
        Vector3d pos_at_t = position + std::max((time_point - PCDstartTime), 0.0) * velocity;
        double dist = (node->coord.head<3>() - pos_at_t).norm();
        min_dynamic_dist = min(min_dynamic_dist, dist);
    }
    double hashing_dist = queryDynamicObstacles(node->coord);

    if(min_dynamic_dist > map_dist)
    {
        node->closest_static = true;
    }
    else
    {
        node->closest_dynamic = true;
    }
    std::cout<<"[path radius check] coord: "<<node->coord.transpose()<<" static dist: "<<map_dist<<" dynamic_dist: "<<min_dynamic_dist<<" hashing dist: "<<hashing_dist<<" radius: "<<node->radius<<std::endl;
}

void safeRegionRrtStarDynamic::clearBranchW(NodePtr_dynamic node_delete) {
    if (!node_delete) {
        cerr << "[Error] Node to delete is null in clearBranchW()" << endl;
        return;
    }
    
    if (!node_delete->valid) return;

    for (auto child : node_delete->nxtNode_ptr) {
        if (!child || !child->valid) continue;

        if (!child->best) {
            if (find(invalidSet.begin(), invalidSet.end(), child) == invalidSet.end()) {
                invalidSet.push_back(child);
            }
            child->valid = false;
            clearBranchW(child);
        }
    }
}

void safeRegionRrtStarDynamic::clearBranchS(NodePtr_dynamic node_delete) {
    if (!node_delete) {
        cerr << "[Error] Node to delete is null in clearBranchS()" << endl;
        return;
    }
    
    if (!node_delete->valid) return;

    for (auto child : node_delete->nxtNode_ptr) {
        if (!child) continue;
        
        if (child->valid) {
            if (find(invalidSet.begin(), invalidSet.end(), child) == invalidSet.end()) {
                invalidSet.push_back(child);
            }
            child->valid = false;
        }
        clearBranchS(child);
    }
}

void safeRegionRrtStarDynamic::treePrune(NodePtr_dynamic newPtr) {     
    NodePtr_dynamic ptr = newPtr;
    if(ptr->g + ptr->f > best_distance) {
        ptr->invalid_in_prune = true;
        ptr->valid = false;
        invalidSet.push_back(ptr);
        clearBranchS(ptr);
    }    
}

void safeRegionRrtStarDynamic::removeInvalid4d() {     
    vector<NodePtr_dynamic> UpdateNodeList;
    node_raw_to_shared_map.clear();
    vector<NodePtr_dynamic> UpdateEndList;
    kd_clear(kdTree_);
    
    for(auto nodeptr : NodeList) {
        if(nodeptr && nodeptr->valid) {
            float pos[4] = {
                (float)nodeptr->coord(0), 
                (float)nodeptr->coord(1), 
                (float)nodeptr->coord(2), 
                (float)nodeptr->coord(3)
            };
            kd_insert4f(kdTree_, pos[0], pos[1], pos[2], pos[3], nodeptr.get());
            UpdateNodeList.push_back(nodeptr);
            node_raw_to_shared_map[nodeptr.get()] = nodeptr;
            if(checkEnd(nodeptr)) {
                UpdateEndList.push_back(nodeptr);
            }
        }
    }

    NodeList = move(UpdateNodeList);
    EndList = move(UpdateEndList);

    auto invalidNodes = invalidSet;
    invalidSet.clear();
    
    for(auto nodeptr : invalidNodes) {
        if(!nodeptr) continue;

        if(nodeptr->preNode_ptr) {
            auto& children = nodeptr->preNode_ptr->nxtNode_ptr;
            
            children.erase(
                remove(children.begin(), children.end(), nodeptr),
                children.end()
            );
        }

        for(auto childptr : nodeptr->nxtNode_ptr) {
            if(childptr && childptr->valid && childptr->preNode_ptr) {
                childptr->preNode_ptr = nullptr;
            }
        }
        
        nodeptr->nxtNode_ptr.clear();
    }
}

Eigen::Vector4d safeRegionRrtStarDynamic::getRootCoords() {
    return root_node->coord;
}

void safeRegionRrtStarDynamic::resetRoot(int root_index) {   
    if(root_index == -1) {
        path_exist_status = false;
        return;
    }
    NodePtr_dynamic target_node = PathList[root_index];
    NodePtr_dynamic lstNode = PathList.front();
    if (!lstNode || !target_node) {
        path_exist_status = false;
        return;
    }

    pair<double, double> ST_pair = getDisMetric(target_node->coord, lstNode->coord);

    if(ST_pair.first < lstNode->radius) {
        global_navi_status = true;
        return;
    }

    double cost_reduction = 0;
    commit_root = target_node->coord;
    vector<NodePtr_dynamic> cutList;
    
    for(auto nodeptr:NodeList) {
        nodeptr->best = false;
    }

    bool delete_root = false;
    double min_dist = INFINITY;

    if (root_node == target_node) {
        return;
    }
    if(target_node && root_index < PathList.size()) {
        PathList[root_index]->best = true;
        PathList[root_index]->preNode_ptr = nullptr;
        cost_reduction = PathList[root_index]->g;
        PathList[root_index]->g = 0.0;
        root_node = PathList[root_index];
    }
    if(!target_node) {
        path_exist_status = false;
        return;
    }
    for(int i=root_index+1; i<PathList.size(); i++) {
        PathList[i]->best = false;
        PathList[i]->valid = false;
        cutList.push_back(PathList[i]);
    }
    // std::cout<<"before node cleanup: "<<std::endl;
    for(auto node : NodeList)
    {
        if(node->valid && node->coord[3] < target_node->coord[3])
        {
            node->best = false;
            node->valid = false;
            cutList.push_back(node);
        }
    }
    // std::cout<<"after node cleanup: "<<std::endl;
    solutionUpdate(cost_reduction, target_node->coord);

    for(auto nodeptr:cutList) {
        invalidSet.push_back(nodeptr);
        clearBranchW(nodeptr);
    }

    removeInvalid4d();
}

void safeRegionRrtStarDynamic::solutionUpdate(double cost_reduction, Vector4d target) {
    Vector3d target3d = target.head<3>();

    for(auto nodeptr: NodeList) {
        nodeptr->g -= cost_reduction;
    }
    min_distance = getDis(target3d, end_pt) * ((1 - weightT) + weightT/max_vel);
    
    translation_inf = (target3d + end_pt) / 2.0;

    Eigen::Vector3d xtf, ytf, ztf, downward(0, 0, -1);
    xtf = (target3d - translation_inf).normalized();
    ytf = xtf.cross(downward).normalized();
    ztf = xtf.cross(ytf);

    rotation_inf.col(0) = xtf;
    rotation_inf.col(1) = ytf;
    rotation_inf.col(2) = ztf;

    best_distance -= cost_reduction;

    double extreme_expected_time = 4 * (getDis(target.head<3>(), end_pt) / max_vel);
    rand_idx = uniform_real_distribution<double>(target[3], target[3] + extreme_expected_time);
}

void safeRegionRrtStarDynamic::updateHeuristicRegion(NodePtr_dynamic update_end_node) {   
    // double update_cost = update_end_node->g + (getDis(update_end_node->coord.head<3>(), end_pt) + getDis(root_node->coord.head<3>(), commit_root.head<3>()))*((1-weightT) + weightT/max_vel);
    // double update_cost = update_end_node->g + getDis(update_end_node->coord.head<3>(), end_pt)*((1 - weightT) + weightT / max_vel);
    double spatial_remain = (end_pt - update_end_node->coord.head<3>()).norm();
    double heuristic_cost = (1 - weightT) * spatial_remain + weightT * (spatial_remain / max_vel);
    double update_cost = update_end_node->g + heuristic_cost;

    if (update_cost + 1e-6 < min_distance) {
        update_cost = min_distance;
    }
    if(update_cost < best_distance) {
        // std::cout<<"[update heuristic debug] update_cost: "<<update_cost<<" best distance: "<<best_distance<<" min_distance: "<<min_distance<<" end_node->g: "<<update_end_node->g<<std::endl;
  
        best_distance = update_cost;
        if(best_distance < min_distance)
        {
            NodePtr_dynamic temp_node = update_end_node;
            while(temp_node != NULL)
            {
                std::cout<<"FUBAR Node coords: "<<temp_node->coord.transpose()<<std::endl;
                std::cout<<"FUBAR Node g: "<<temp_node->g<<std::endl;

                temp_node = temp_node->preNode_ptr;
            }
            throw std::runtime_error("FUBAR code spotted!");
        }
        elli_l = best_distance;
        elli_s = sqrt(best_distance * best_distance - min_distance * min_distance);

        if(inform_status) {
            for(auto ptr:NodeList)
                ptr->best = false;
        }

        NodePtr_dynamic ptr = update_end_node;
        while(ptr) {  
            ptr->best = true;
            ptr = ptr->preNode_ptr;
        }

        best_end_ptr = update_end_node;
    }    
}

Vector4d safeRegionRrtStarDynamic::genSample4d() {
    Vector4d pt4d;
    double bias = rand_bias(eng);

    if (bias <= goal_ratio) {
        pt4d.head<3>() = end_pt;
        pt4d(3) = (rand_u(eng) + 1)*(getDis(end_pt, root_node->coord.head<3>()))/max_vel;
        return pt4d;
    }

    if (!inform_status) {
        // Original uniform sampling
        if (bias > goal_ratio && bias <= (goal_ratio + inlier_ratio)) {
            pt4d(0) = rand_x_in(eng);
            pt4d(1) = rand_y_in(eng);
            pt4d(2) = rand_z_in(eng);
        } else {
            pt4d(0) = rand_x(eng);
            pt4d(1) = rand_y(eng);
            pt4d(2) = rand_z(eng);
        }
        pt4d(3) = rand_idx(eng);
        if (!std::isfinite(pt4d[0]) || !std::isfinite(pt4d[1]) || !std::isfinite(pt4d[2]))
        {
            std::cerr<<"infinite sample generation error in NOT inform status: "<<std::endl;
        } 

    } else {
        // --- Improved 4D Ellipsoidal Sampling ---
        double us = rand_u(eng);       // Uniform [0,1]
        double vs = rand_v(eng);       // Uniform [0,1]
        double phis = rand_phi(eng);   // Uniform [0, 2π]
        double time_s = rand_u(eng);   // New: Time component

        // Scale spatial axes (x,y,z)
        double as = elli_l / 2.0 * cbrt(us);
        double bs = elli_s / 2.0 * cbrt(us);
        double thetas = acos(1 - 2 * vs);

        // Scale time axis (t)
        double max_time = elli_l / max_vel;  // Time scaling based on max velocity
        double ts = max_time * time_s;       // Time within ellipsoid bounds

        // Convert to Cartesian coordinates (x,y,z,t)
        Vector4d pt_local;
        pt_local(0) = as * sin(thetas) * cos(phis);
        pt_local(1) = bs * sin(thetas) * sin(phis);
        pt_local(2) = bs * cos(thetas);
        pt_local(3) = ts;  // Time component

        if (!std::isfinite(pt_local[0]) || !std::isfinite(pt_local[1]) || !std::isfinite(pt_local[2]))
        {
            throw std::runtime_error("Infinite sample generation error in inform status!");
        }
        // Rotate and translate spatial dimensions (x,y,z)
        pt4d.head<3>() = rotation_inf * pt_local.head<3>() + translation_inf;
        pt4d(3) = root_node->coord(3) + pt_local(3);  // Offset from root time

        // Clamp to bounds
        pt4d(0) = clamp(pt4d(0), x_l, x_h);
        pt4d(1) = clamp(pt4d(1), y_l, y_h);
        pt4d(2) = clamp(pt4d(2), z_l, z_h);
        pt4d(3) = max(pt4d(3), root_node->coord(3));  // Ensure time doesn't go backwards
    }

    return pt4d;
}
inline NodePtr_dynamic safeRegionRrtStarDynamic::genNewNode(Vector4d & pt_sample, NodePtr_dynamic node_nearest_ptr) {
    pair<double, double> ST_dis = getDisMetric(node_nearest_ptr->coord, pt_sample);

    double dis = ST_dis.first;
    double dis_T = ST_dis.second;
    // if(dis_T < 0.0) std::cout<<"[invalid dT in genNewNode]"<<std::endl;
    Vector4d center;
    if(dis_T > 2*delta_t) {
        center[3] = node_nearest_ptr->coord[3] + 2*delta_t;
        dis_T = 2*delta_t;
    }
    else if(dis_T < delta_t/2) {
        center[3] = node_nearest_ptr->coord[3] + delta_t/2;
        dis_T = delta_t/2;
    }
    if (dis > max_vel * 0.9 * dis_T) {
        Eigen::Vector3d direction = (pt_sample.head<3>() - node_nearest_ptr->coord.head<3>()).normalized();
        center.head<3>() = node_nearest_ptr->coord.head<3>() + direction * (max_vel * 0.9 * dis_T);
    } else {
        center.head<3>() = pt_sample.head<3>();
    }
    // auto t1 = std::chrono::steady_clock::now();

    double radius_ = radiusSearch(center);
    // auto t2 = std::chrono::steady_clock::now();
    // auto elapsed_radiussearch = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()*1e-6;
    // std::cout<<"[gen new node debug] elapsed radius search: "<<elapsed_radiussearch<<std::endl;
    double h_dis_ = getDis(center.head<3>(), end_pt);
    h_dis_ = (1-weightT)*h_dis_ + weightT*h_dis_/max_vel;
    auto node_new_ptr = make_shared<Node_dynamic>(center, radius_, INFINITY, h_dis_);
    return node_new_ptr;
}

bool safeRegionRrtStarDynamic::checkTrajPtCol(Vector4d & pt) {     
    return (radiusSearch(pt) < 0.1);
}

inline bool safeRegionRrtStarDynamic::checkEnd(NodePtr_dynamic ptr) {    
    double distance = getDis(ptr->coord.head<3>(), end_pt);
    return (distance < max_vel*2*delta_t);
}

inline NodePtr_dynamic safeRegionRrtStarDynamic::findNearestVertex4d(Vector4d & pt) {
    float pos[4] = {(float)pt(0), (float)pt(1), (float)pt(2), (float)pt(3)};
    kdres * nearest = kd_nearest4f(kdTree_, pos[0], pos[1], pos[2], pos[3]);

    // Get the raw pointer from KD-tree
    Node_dynamic* raw_ptr = static_cast<Node_dynamic*>(kd_res_item_data(nearest));
    
    // We need to ensure we don't create a new shared_ptr that would try to delete the node
    // Since the node is already owned by another shared_ptr in NodeList
    NodePtr_dynamic node_nearest_ptr;
    // for (const auto& node : NodeList) {
    //     if (node.get() == raw_ptr) {
    //         node_nearest_ptr = node;
    //         break;
    //     }
    // }
    auto it = node_raw_to_shared_map.find(raw_ptr);
    if (it != node_raw_to_shared_map.end())
    {
        node_nearest_ptr = it->second;
    }
    
    kd_res_free(nearest);
    return node_nearest_ptr;
}

inline int safeRegionRrtStarDynamic::checkNodeRelation(NodePtr_dynamic node_parent, NodePtr_dynamic node_child) {
    int status;
    auto ST_pair = getDisMetric(node_parent, node_child);
    double dis = ST_pair.first;

    if((dis + node_parent->radius) == node_child->radius)
        status = 1;
    else if((dis + 0.1) < 0.95 * (node_parent->radius + node_child->radius))
        status = -1;
    else
    {
        if(_bkup) status = -1;
        else status = 0;
    }    
    return status;
}

void safeRegionRrtStarDynamic::treeRewire(NodePtr_dynamic node_new_ptr, NodePtr_dynamic node_nearst_ptr) {     
    NodePtr_dynamic newPtr = node_new_ptr;      
    NodePtr_dynamic nearestPtr = node_nearst_ptr;

    float range = sqrt(pow(newPtr->radius * 2.0f,2) + pow(2.0 * delta_t, 2));
    float pos[4] = {(float)newPtr->coord(0), (float)newPtr->coord(1), (float)newPtr->coord(2), (float)newPtr->coord(3)};
    struct kdres *presults = kd_nearest_range4f(kdTree_, pos[0], pos[1], pos[2], pos[3], range);
    const double t0 = root_node->coord(3);
    vector<NodePtr_dynamic> nearPtrList;
    bool isInvalid = false;
    while(!kd_res_end(presults)) 
    {
        Node_dynamic* raw_ptr = static_cast<Node_dynamic*>(kd_res_item_data(presults));
        
        // Find the existing shared_ptr that owns this node
        NodePtr_dynamic nearPtr;
        auto it = node_raw_to_shared_map.find(raw_ptr);
        if (it != node_raw_to_shared_map.end())
        {
            nearPtr = it->second;
        }

        if (!nearPtr) {
            kd_res_next(presults);
            continue;
        }

        auto ST_pair = getDisMetric(nearPtr, newPtr);
        double dis = (1-weightT)*ST_pair.first + weightT*ST_pair.second;
        if(ST_pair.second <= delta_t/2 || ST_pair.second > 2*delta_t || ST_pair.first > max_vel*ST_pair.second) dis = INFINITY;
        int res = checkNodeRelation(nearPtr, newPtr);
        nearPtr->rel_id = res;
        nearPtr->rel_dis = dis;

        if(res == 1) {
            newPtr->invalid_in_rewire = true;
            newPtr->valid = false;
            isInvalid = true;
            nearPtrList.push_back(nearPtr);
            break;
        }
        else {
            if(ST_pair.second > delta_t/2) {
                nearPtrList.push_back(nearPtr);
            }
            kd_res_next(presults);
        }
    }
    kd_res_free(presults);

    if(isInvalid) {
        for(auto nodeptr: nearPtrList) {
            nodeptr->rel_id = -2;
            nodeptr->rel_dis = -1.0;
        }
        return;
    }
    auto st_dis = getDisMetric(nearestPtr, newPtr);

    double min_cost = nearestPtr->g + (1-weightT)*st_dis.first + weightT*st_dis.second;
    if(st_dis.second <= delta_t/2 or st_dis.second > 2*delta_t || st_dis.first > max_vel*st_dis.second) min_cost = INFINITY;
    newPtr->preNode_ptr = nearestPtr;
    newPtr->g = min_cost;
    nearestPtr->nxtNode_ptr.push_back(newPtr);
    NodePtr_dynamic lstParentPtr = nearestPtr;

    vector<NodePtr_dynamic> nearVertex;

    for(auto nodeptr:nearPtrList) {
        NodePtr_dynamic nearPtr = nodeptr;
        if(std::isfinite(nearPtr->g) && nearPtr->coord[3] >= t0)
        {
            int res = nearPtr->rel_id;
            double dis = nearPtr->rel_dis;
            double cost = nearPtr->g + dis;
            
            if(res == -1) {
                if(cost < min_cost) {
                    min_cost = cost;
                    newPtr->preNode_ptr = nearPtr;
                    newPtr->g = min_cost;
                    lstParentPtr->nxtNode_ptr.pop_back();
                    lstParentPtr = nearPtr;
                    lstParentPtr->nxtNode_ptr.push_back(newPtr);
                }
                nearVertex.push_back(nearPtr);
            }      
            nearPtr->rel_id = -2;
            nearPtr->rel_dis = -1.0;
        }
    }

    for(int i = 0; i < int(nearVertex.size()); i++) {
        NodePtr_dynamic nodeptr = nearVertex[i];
        NodePtr_dynamic nearPtr = nodeptr;
        if(nearPtr->valid == false) continue;

        auto st_pair = getDisMetric(newPtr, nearPtr);
        double dis = (1-weightT)*st_pair.first + weightT*st_pair.second;
        if(st_pair.second <= delta_t/2 || st_pair.second > 2*delta_t || st_pair.first > max_vel*st_pair.second) dis = INFINITY;
        double cost = dis + newPtr->g;
        
        if(cost < nearPtr->g) {
            if(isSuccessor(nearPtr, newPtr->preNode_ptr)) 
                continue;
              
            if(!nearPtr->preNode_ptr) {
                nearPtr->preNode_ptr = newPtr;
                nearPtr->g = cost;
            }
            else {
                NodePtr_dynamic lstNearParent = nearPtr->preNode_ptr;
                nearPtr->preNode_ptr = newPtr;
                nearPtr->g = cost;
                nearPtr->change = true;
                vector<NodePtr_dynamic> child = lstNearParent->nxtNode_ptr;
                    
                lstNearParent->nxtNode_ptr.clear();
                for(auto ptr: child) {
                    if(ptr->change == true) continue;
                    else lstNearParent->nxtNode_ptr.push_back(ptr);
                }
                nearPtr->change = false;
            }
            newPtr->nxtNode_ptr.push_back(nearPtr);
        }
    }
    // if(newPtr->g < 0 && !std::isfinite(newPtr->g))
    // {
    //     std::cout<<"################## trouble in rewiring: "<<std::endl;
    //     std::cout<<"coords: "<<newPtr->coord.transpose()<<std::endl;
    //     std::cout<<"cost: "<<newPtr->g<<std::endl;
    //     auto temp_node = newPtr;
    //     while(temp_node)
    //     {
    //         std::cout<<"debug coords: "<<temp_node->coord.transpose()<<std::endl;
    //         std::cout<<"debug cost: "<<temp_node->g<<std::endl;
    //         temp_node = temp_node->preNode_ptr;
    //     }
    //     throw std::runtime_error("Negative cost values!");

    // }
}

void safeRegionRrtStarDynamic::recordNode(NodePtr_dynamic new_node) {
    if(new_node->preNode_ptr) {
        if(new_node->coord[3] <= new_node->preNode_ptr->coord[3]) {
            cerr << "[record node debug] temporal constraint violation" << endl;
        }
    }
    NodeList.push_back(new_node);
    node_raw_to_shared_map[new_node.get()] = new_node;
}

void safeRegionRrtStarDynamic::tracePath() {   
    vector<NodePtr_dynamic> feasibleEndList;
    for(auto endPtr: EndList) {
        if((checkValidEnd(endPtr) == false) || (checkEnd(endPtr) == false) || (endPtr->valid == false)) {
            continue;
        }
        else
            feasibleEndList.push_back(endPtr);
    }

    if(int(feasibleEndList.size()) == 0) {
        path_exist_status = false;
        best_distance = INFINITY;
        inform_status = false;
        EndList.clear();
        Path = Eigen::MatrixXd::Identity(3,3);
        Radius = Eigen::VectorXd::Zero(3);
        return;
    }

    EndList = feasibleEndList;

    best_end_ptr = feasibleEndList[0];
    double best_cost = INFINITY;
    for(auto nodeptr: feasibleEndList) {   
        double dis_1 = getDis(nodeptr->coord.head<3>(), end_pt)*((1-weightT) + weightT/max_vel);
        auto d2 = getDisMetric(root_node->coord, commit_root);
        double dis_2 = (1-weightT)*d2.first + weightT*d2.second;
        double cost = (nodeptr->g + dis_1 + dis_2);
        if(cost < best_cost) {
            best_end_ptr = nodeptr;
            best_cost = cost;
            best_distance = best_cost;
        }
    }

    NodePtr_dynamic ptr = best_end_ptr;
    int idx = 0;
    PathList.clear();

    while(ptr) {
        PathList.push_back(ptr);
        ptr = ptr->preNode_ptr;
        idx++;
    }

    Path.resize(idx, 4);
    Radius.resize(idx);
    idx = 1;
    
    ptr = best_end_ptr;
    while(ptr) {      
        Path.row(Path.rows() - idx) = ptr->coord;
        Radius(Radius.size() - idx) = ptr->radius;
        // std::cout<<"[trace path debug] point: "<<ptr->coord.transpose()<<" radius: "<<ptr->radius<<std::endl;
        ptr = ptr->preNode_ptr;
        idx++;
    }

    path_exist_status = true;
}

void safeRegionRrtStarDynamic::treeDestruct() {
    if(kdTree_ != NULL) {
        kd_free(kdTree_);
    }
    NodeList.clear();
    node_raw_to_shared_map.clear();
}

void safeRegionRrtStarDynamic::SafeRegionExpansion(double time_limit, double root_node_time, bool bkup) {   
    auto time_bef_expand = chrono::steady_clock::now();
    kdTree_ = kd_create(4);
    commit_root.head<3>() = start_pt;
    Vector4d start_pt_4d;
    start_pt_4d << start_pt, root_node_time;
    root_node = make_shared<Node_dynamic>(start_pt_4d, radiusSearch(start_pt_4d), 0.0, min_distance);
    rand_idx = uniform_real_distribution<double>(root_node_time, root_node_time + 4*getDis(start_pt, end_pt)/max_vel);

    recordNode(root_node);
    _bkup = bkup;
    float pos4d[4] = {(float)root_node->coord(0), (float)root_node->coord(1), (float)root_node->coord(2), (float)root_node->coord(3)};
    kd_insertf(kdTree_, pos4d, root_node.get());
    int iter_count;

    for(iter_count = 0; iter_count < max_samples; iter_count++) {     
        auto time_in_expand = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(time_in_expand - time_bef_expand).count()*0.001;
        if(elapsed > time_limit) {
            break;
        }
        
        Vector4d pt_sample = genSample4d();
        if (!std::isfinite(pt_sample[0]) || !std::isfinite(pt_sample[1]) || !std::isfinite(pt_sample[2])) 
        {
            std::cerr << "Invalid point coordinates in gen sample: " 
                      << pt_sample.transpose() << std::endl;
            return;
        }
        NodePtr_dynamic node_nearest_ptr = findNearestVertex4d(pt_sample);
        
        if(!node_nearest_ptr->valid || !node_nearest_ptr || node_nearest_ptr->coord[3] < root_node->coord[3]) {
            if(node_nearest_ptr->coord[3] < root_node->coord[3])
            {
                node_nearest_ptr->valid = false;
            }
            continue;
        }
        if(node_nearest_ptr->coord(3) >= pt_sample(3)) 
        {
            pt_sample(3) = node_nearest_ptr->coord(3) + static_cast<float>(uniform_real_distribution<double>(delta_t/2, 2 * delta_t)(eng));
        }

        NodePtr_dynamic node_new_ptr = genNewNode(pt_sample, node_nearest_ptr);
        if (!std::isfinite(node_new_ptr->coord[0]) || !std::isfinite(node_new_ptr->coord[1]) || !std::isfinite(node_new_ptr->coord[2]) || !std::isfinite(node_new_ptr->coord[3])) 
        {
            std::cerr << "Invalid point coordinates in gen new node: " 
                      << node_new_ptr->coord.transpose() << std::endl;
            return;
        }
        if(node_new_ptr->coord(2) < (z_l + safety_margin) || node_new_ptr->radius < safety_margin) {
            continue;
        }
        if(node_new_ptr->coord[3] <= node_nearest_ptr->coord[3]) {
            continue;
        }
        treeRewire(node_new_ptr, node_nearest_ptr);  
        if (std::isinf(node_new_ptr->g) && node_new_ptr->g < 0) 
        {
            continue;
            // throw std::runtime_error("Negative infinity g detected in expansion!");
        }

        if(!node_new_ptr->valid) {
            continue;
        }

        if(checkEnd(node_new_ptr)) {
            if(!inform_status) best_end_ptr = node_new_ptr;
            if(_bkup)
            {
                EndList.push_back(node_new_ptr);
                updateHeuristicRegion(node_new_ptr);
                inform_status = true;
                pos4d[0] = (float)node_new_ptr->coord(0);
                pos4d[1] = (float)node_new_ptr->coord(1);
                pos4d[2] = (float)node_new_ptr->coord(2);
                pos4d[3] = (float)node_new_ptr->coord(3);
                kd_insertf(kdTree_, pos4d, node_new_ptr.get());

                recordNode(node_new_ptr);
                treePrune(node_new_ptr);      
                break;
            }
            EndList.push_back(node_new_ptr);
            updateHeuristicRegion(node_new_ptr);
            inform_status = true;  
        }
        
        pos4d[0] = (float)node_new_ptr->coord(0);
        pos4d[1] = (float)node_new_ptr->coord(1);
        pos4d[2] = (float)node_new_ptr->coord(2);
        pos4d[3] = (float)node_new_ptr->coord(3);
        kd_insertf(kdTree_, pos4d, node_new_ptr.get());

        recordNode(node_new_ptr);
        treePrune(node_new_ptr);      

        if(int(invalidSet.size()) >= cach_size) removeInvalid4d();
    }

    removeInvalid4d();
    tracePath();
}

void safeRegionRrtStarDynamic::SafeRegionRefine( double time_limit )
{   
    /*  Every time the refine function is called, new samples are continuously generated and the tree is continuously rewired, hope to get a better solution  */
    /*  The refine process is mainly same as the initialization of the tree  */
    auto time_bef_refine = std::chrono::steady_clock::now();
    // std::cout<<"in refine loop"<<std::endl;

    float pos[4];
    while( true )
    {     
        auto time_in_refine = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_in_refine - time_bef_refine).count()*0.001;
        if( elapsed > time_limit )
        {
            break;
        } 
        Vector4d pt_sample =  genSample4d();
        NodePtr_dynamic node_nearest_ptr = findNearestVertex4d(pt_sample);
        if(!node_nearest_ptr->valid || !node_nearest_ptr || node_nearest_ptr->coord[3] < root_node->coord[3]) 
        {
            if(node_nearest_ptr != NULL && node_nearest_ptr->coord[3] < root_node->coord[3])
            {
                node_nearest_ptr->valid = false;
            }
            continue;
        }

        if(node_nearest_ptr->coord(3) >= pt_sample(3))
        {
           pt_sample(3) = node_nearest_ptr->coord(3) + static_cast<float>(uniform_real_distribution<double>(delta_t/2, 2 * delta_t)(eng));
        }
        NodePtr_dynamic node_new_ptr  =  genNewNode(pt_sample, node_nearest_ptr); 
        if( node_new_ptr->coord(2) < (z_l + safety_margin) || node_new_ptr->radius < safety_margin )
        {
            continue;
        }
        if(node_new_ptr->coord[3] <= node_nearest_ptr->coord[3])
        {
            continue;
        }
        treeRewire(node_new_ptr, node_nearest_ptr);  
        // if(node_new_ptr->preNode_ptr == root_node)
        // {
        //     std::cout<<"refine debug: child ptr added to root node"<<std::endl;
        // }
        if (std::isinf(node_new_ptr->g) && node_new_ptr->g < 0) 
        {
            continue;
            // throw std::runtime_error("Negative infinity g detected in refine!");
        }
        if( node_new_ptr->valid == false ) continue;
        if(checkEnd( node_new_ptr )){
            if( !inform_status ) 
            {
                best_end_ptr = node_new_ptr;
            }            
            EndList.push_back(node_new_ptr);
            updateHeuristicRegion(node_new_ptr);
            inform_status = true;  
        }
        // std::cout<<"[refine seg debug 17]"<<std::endl;
        pos[0] = (float)node_new_ptr->coord(0);
        pos[1] = (float)node_new_ptr->coord(1);
        pos[2] = (float)node_new_ptr->coord(2);
        pos[3] = (float)node_new_ptr->coord(3);
        kd_insert4f(kdTree_, pos[0], pos[1], pos[2], pos[3], node_new_ptr.get());
        recordNode(node_new_ptr);
        treePrune(node_new_ptr);      
        if( int(invalidSet.size()) >= cach_size) removeInvalid4d();
  }
    // std::cout<<"[refine seg debug 2]"<<std::endl;
    removeInvalid4d();  
    // std::cout<<"[refine seg debug 3]"<<std::endl;  
    tracePath();

}

void safeRegionRrtStarDynamic::SafeRegionEvaluate( double time_limit )
{   
    /*  The re-evaluate of the RRT*, find the best feasble path (corridor) by lazy-evaluzation  */
    if(path_exist_status == false){
        // std::cout<<"[path re-evaluate] no path exists. "<<std::endl;
        return;
    }
    
    auto time_bef_evaluate = std::chrono::steady_clock::now();

    vector< pair<Vector4d, double> > fail_node_list;
    while(true)
    {   
        for(int i = 0; i < int(PathList.size()); i++ ) 
        {   
            NodePtr_dynamic ptr = PathList[i];
            NodePtr_dynamic pre_ptr =  ptr->preNode_ptr;

            if( pre_ptr != NULL ){
                double update_radius = checkRadius(ptr->coord);
                int ret = checkNodeUpdate(update_radius, ptr->radius); // -1: not qualified, 0: shrink, 1: no difference, continue
                double old_radius = ptr->radius;
                ptr->radius = update_radius; // the radius of a node may shrink or remain no change, but can not enlarge

                if( ret == -1){   
                    // std::cout<<"ptr invalid in checkNodeUpdate: "<<ptr->coord.transpose()<<"radius: "<<ptr->radius<<std::endl;
                    ptr->valid = false;          // Delete this node
                    invalidSet.push_back(ptr);
                    clearBranchS(ptr);
                    fail_node_list.push_back(make_pair(ptr->coord, old_radius));
                }
                else 
                {   // Test whether the node disconnected with its parent and all children,
                    if( checkNodeRelation( ptr, pre_ptr ) != -1) {
                        // the child is disconnected with its parent   
                        if(ptr->valid == true){
                            double a = checkNodeRelation(ptr, pre_ptr);
                            double b = ptr->coord[3] - pre_ptr->coord[3];
                            // std::cout<<"[evaluate] ptr invalid in checkNodeRelation: "<<ptr->coord.transpose()<<std::endl;
                            // std::cout<<" a: "<<a<<" b: "<<b<<std::endl;
                            auto p = getDisMetric(root_node, ptr);
                            // std::cout<<"root node distance of invalid ptr: "<<p.first<<" : "<<p.second<<std::endl;
                            ptr->valid = false;      
                            invalidSet.push_back(ptr);
                            clearBranchS(ptr);
                            fail_node_list.push_back(make_pair(ptr->coord, old_radius));
                        }
                    } 
                    else
                    {
                        vector<NodePtr_dynamic> childList = ptr->nxtNode_ptr;
                        for(auto nodeptr: childList){  
                            // inspect each child of ptr, to see whether they are still connected 
                            int res = checkNodeRelation( ptr, nodeptr );
                            if( res != -1) 
                            {   
                                // std::cout<<"childptr invalid in checkNodeRelation: "<<nodeptr->coord.transpose()<<std::endl;
                                // std::cout<<"parent: "<<ptr->coord.transpose()<<std::endl;
                                // std::cout<<"res: "<<checkNodeRelation(ptr, nodeptr)<<std::endl;
                                auto p = getDisMetric(ptr, nodeptr);
                                // std::cout<<"node distance: "<<p.first<<" radius child: "<<ptr->radius<<" radius parent: "<<nodeptr->radius<<std::endl;
                                // the child is disconnected with its parent
                                if(nodeptr->valid == true){
                                    nodeptr->valid = false;      
                                    invalidSet.push_back(nodeptr);
                                    clearBranchS(nodeptr);
                                    fail_node_list.push_back(make_pair(nodeptr->coord, nodeptr->radius));
                                }
                            } 
                        }
                    }
                }
            }
        }

        bool isBreak = true;

        for(auto ptr:PathList) 
          isBreak = isBreak && ptr->valid;
        
        auto time_in_evaluate2 = std::chrono::steady_clock::now();

        if(isBreak)
        {
            break;
        }

        vector<NodePtr_dynamic> feasibleEndList;                
        for(auto endPtr: EndList)
        {
            if( (endPtr->valid == false)  || (checkEnd(endPtr) == false) )
                continue;
            else
                feasibleEndList.push_back(endPtr);
        }


        EndList = feasibleEndList;
        auto time_in_evaluate = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_in_evaluate - time_bef_evaluate).count()*0.001;
        if(feasibleEndList.size() == 0 || elapsed > time_limit ){
            path_exist_status = false;
            inform_status = false;
            best_distance = INFINITY;
            break;
        }
        else{
            best_end_ptr = feasibleEndList[0];
            double best_cost = INFINITY;
            for(auto nodeptr: feasibleEndList)
            {   
                auto p_ST = getDisMetric(root_node->coord, commit_root);

                double cost = (nodeptr->g + getDis(nodeptr->coord.head<3>(), end_pt)*((1 - weightT) + weightT/max_vel) + (1 - weightT)*p_ST.first + p_ST.second * weightT);
                if( cost < best_cost ){
                    best_end_ptr  = nodeptr;
                    best_cost = cost;
                    best_distance = best_cost;
                }
            }
        
            PathList.clear();
            NodePtr_dynamic ptrr = best_end_ptr;
            while( ptrr != NULL )
            {
                PathList.push_back( ptrr );
                ptrr = ptrr->preNode_ptr;
            }
        }
    } 
    
    auto time_aft_evaluate = std::chrono::steady_clock::now();
    double repair_limit = time_limit - std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_evaluate - time_bef_evaluate).count()*0.001;

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
            Node_dynamic* raw_ptr = static_cast<Node_dynamic*>(kd_res_item_data(presults));
            
            // Find the existing shared_ptr that owns this node
            NodePtr_dynamic ptr;
            // for (const auto& node : NodeList) {
            //     if (node.get() == raw_ptr) {
            //         ptr = node;
            //         break;
            //     }
            // }
            auto it = node_raw_to_shared_map.find(raw_ptr);
            if (it != node_raw_to_shared_map.end())
            {
                ptr = it->second;
            }
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
            // std::cout<<"[repair debug] after radius search"<<std::endl;
            int ret = checkNodeUpdate(update_radius, ptr->radius);
            ptr->radius = update_radius;
            // std::cout<<"[repair debug] debug 1"<<std::endl;

            if (ret == -1) { // Node became invalid
                ptr->valid = false;
                invalidSet.push_back(ptr);
                clearBranchS(ptr);
                continue;
            }
            // std::cout<<"[repair debug] debug 2"<<std::endl;
            // Check parent connection with 4D distance
            if (pre_ptr && checkNodeRelation(pre_ptr, ptr) != -1) {
                ptr->valid = false;
                invalidSet.push_back(ptr);
                clearBranchS(ptr);
                continue;
            }
            // std::cout<<"[repair debug] debug 3"<<std::endl;
            // Check child connections
            for (auto childptr : ptr->nxtNode_ptr) {
                // Verify temporal ordering
                if (childptr->coord[3] < ptr->coord[3]) {
                    childptr->valid = false;
                    invalidSet.push_back(childptr);
                    clearBranchS(childptr);
                    continue;
                }
                // std::cout<<"[repair debug] debug 3a"<<std::endl;
                if (checkNodeRelation(ptr, childptr) != -1) {
                    childptr->valid = false;
                    invalidSet.push_back(childptr);
                    clearBranchS(childptr);
                }
                // std::cout<<"[repair debug] debug 3b"<<std::endl;
            }
        }
        // std::cout<<"[repair debug] debug 4"<<std::endl;
        kd_res_free(presults);
        // std::cout<<"[repair debug] debug 5"<<std::endl;
    }
    // std::cout<<"[repair debug] debug 6"<<std::endl;

    removeInvalid4d();
    // std::cout<<"[repair debug] debug 7"<<std::endl;

}