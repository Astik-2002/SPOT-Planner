#ifndef _DATA_TYPE_DYNAMIC_
#define _DATA_TYPE_DYNAMIC_

#include <Eigen/Eigen>
#include <unordered_map>  
#include <memory>  // Add this for smart pointers
#include <vector>
using namespace std;

struct Node_dynamic;
typedef shared_ptr<Node_dynamic> NodePtr_dynamic;  // Changed to shared_ptr

struct Node_dynamic
{     
      Eigen::Vector4d coord;
      float radius; // radius of this node

      bool valid;
      bool best;
      bool change;
      bool invalid_in_prune;
      bool invalid_in_rewire;

      // temporary variables, only for speedup the tree rewire procedure
      int   rel_id;
      float rel_dis;
      bool closest_static;
      bool closest_dynamic;
      NodePtr_dynamic preNode_ptr;
      vector<NodePtr_dynamic> nxtNode_ptr;  // Changed to vector of shared_ptr

      float g; // total cost of the shortest path from this node to the root
      float f; // heuristic value of the node to the target point
      
      Node_dynamic(Eigen::Vector4d coord_, float radius_, float g_, float f_)
      {		
            coord  = coord_;
            radius = radius_;
            g      = g_;  
            f      = f_;
            
            rel_id  = -2;   // status undefined
            rel_dis = -1.0; // distance undifined

            valid  = true;
            best   = false; 
            change = false;
            invalid_in_prune = false;
            invalid_in_rewire = false;
            closest_static = false;
            closest_dynamic = false;
            preNode_ptr = nullptr;  // Changed from NULL to nullptr
            nxtNode_ptr.clear();
      }

      Node_dynamic() = default;
      ~Node_dynamic() = default;
};

#endif