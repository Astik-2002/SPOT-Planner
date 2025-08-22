#ifndef _DATA_TYPE_SHARED_
#define _DATA_TYPE_SHARED_

#include <Eigen/Eigen>
#include <unordered_map>  
#include <memory>
#include <vector>
using namespace std;

struct Node_shared;
typedef shared_ptr<Node_shared> NodePtr_shared;  // Changed to shared_ptr

struct Node_shared
{     
      Eigen::Vector3d coord;
      float radius; // radius of this node

      bool valid;
      bool best;
      bool change;
      bool invalid_in_prune;
      bool invalid_in_rewire;

      // temporary variables, only for speedup the tree rewire procedure
      int   rel_id;
      float rel_dis;
      NodePtr_shared preNode_ptr;
      vector<NodePtr_shared> nxtNode_ptr;  // Changed to vector of shared_ptr

      float g; // total cost of the shortest path from this node to the root
      float f; // heuristic value of the node to the target point
      
      Node_shared(Eigen::Vector3d coord_, float radius_, float g_, float f_)
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
            preNode_ptr = nullptr;  // Changed from NULL to nullptr
            nxtNode_ptr.clear();
      }

      Node_shared() = default;
      ~Node_shared() = default;
};

#endif