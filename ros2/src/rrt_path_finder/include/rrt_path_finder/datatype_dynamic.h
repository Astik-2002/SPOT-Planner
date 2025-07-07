#ifndef _DATA_TYPE_DYNAMIC_
#define _DATA_TYPE_DYNAMIC_

#include <Eigen/Eigen>
#include <unordered_map>  
using namespace std;

struct Node_dynamic;
typedef Node_dynamic * NodePtr_dynamic;

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

      NodePtr_dynamic preNode_ptr;
      vector<NodePtr_dynamic> nxtNode_ptr;

      float g; // total cost of the shortest path from this node to the root
      float f; // heuristic value of the node to the target point
      
      Node_dynamic( Eigen::Vector4d coord_, float radius_, float g_, float f_)
      {		
		coord  = coord_;
		radius = radius_;
		g      = g_;  
		f      = f_;
		
		rel_id  = - 2;   // status undefined
		rel_dis = - 1.0; // distance undifined

		valid  = true;
		best   = false; 
      	change = false;
            invalid_in_prune = false;
            invalid_in_rewire = false;

		preNode_ptr = NULL;
      	nxtNode_ptr.clear();
      }

      Node_dynamic(){}
      ~Node_dynamic(){}
};

#endif