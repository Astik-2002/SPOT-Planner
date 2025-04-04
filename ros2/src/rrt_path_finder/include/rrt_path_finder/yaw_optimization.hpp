#include "lbfgs.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>
#include "spline.h"
#include <bits/stdc++.h>
#include "non_uniform_bspline.hpp"

class yawOptimizer
{
    public:
        tk::spline generate_simple_spline(std::vector<double> psi_vec, std::vector<double> time_vec)
        {
            if(is_sorted(time_vec.begin(), time_vec.end()))
            {
                tk::spline yaw_spline(time_vec,psi_vec);
                return yaw_spline;
            }
            else
            {
                stable_sort(time_vec.begin(), time_vec.end());
                tk::spline yaw_spline(time_vec,psi_vec);
                return yaw_spline;
            }
        }

        static inline double cost_functional(void* ptr, const Eigen::VectorXd &x, Eigen::VectorXd &g)
        {
            void **dataPtrs = (void **)ptr;
            const std::vector<double> &des_yaw_vec = *((std::vector<double> *)(dataPtrs[0]));
            const std::vector<int> &wp_idx_vec = *((std::vector<int> *)(dataPtrs[1]));
            const double lambda_wp = *((const double *)(dataPtrs[2]));
            const double lambda_smoothness = *((const double *)(dataPtrs[3]));
            const double lambda_endpoint = 0.001;

            double cost = 0.0;
            g.setZero();
            int num_wp = des_yaw_vec.size();

            // Start and end cost:
            double dq_ini = x[0] - des_yaw_vec.front();
            double dq_fin = x[x.size() - 1] - des_yaw_vec.back();
            cost += lambda_endpoint * (dq_ini * dq_ini + dq_fin * dq_fin);
            g[0] += lambda_endpoint * 2 * dq_ini;
            g[x.size() - 1] += lambda_endpoint * 2 * dq_fin;

            // Waypoint cost:
            for (int i = 0; i < num_wp; ++i)
            {
                double des_yaw = des_yaw_vec[i];
                int wp_idx = wp_idx_vec[i];
                
                if (wp_idx + 2 < x.size()) {
                    double q1 = x[wp_idx];
                    double q2 = x[wp_idx + 1];
                    double q3 = x[wp_idx + 2];
                    double dq = (q1 + 4 * q2 + q3) / 6.0 - des_yaw;
                    cost += lambda_wp * dq * dq;
                    g[wp_idx] += lambda_wp * dq * 0.33;
                    g[wp_idx + 1] += lambda_wp * dq * 1.33;
                    g[wp_idx + 2] += lambda_wp * dq * 0.33;
                }
            }

            // Smoothness cost:
            for (size_t i = 1; i < x.size() - 1; ++i)
            {
                double q_prev = x[i - 1];
                double q_curr = x[i];
                double q_next = x[i + 1];
                double d2q = q_next - 2 * q_curr + q_prev;  // Second derivative
                cost += lambda_smoothness * d2q * d2q;
                g[i - 1] += lambda_smoothness * 2 * d2q;
                g[i] += lambda_smoothness * (-4 * d2q);
                g[i + 1] += lambda_smoothness * 2 * d2q;
            }

            return cost;
        }

        static inline double cost_functional_fixed(void* ptr, const Eigen::VectorXd &x, Eigen::VectorXd &g)
        {
            void **dataPtrs = (void **)ptr;
            const std::vector<double> &des_yaw_vec = *((std::vector<double> *)(dataPtrs[0]));
            const std::vector<int> &wp_idx_vec = *((std::vector<int> *)(dataPtrs[1]));
            const double lambda_wp = *((const double *)(dataPtrs[2]));
            const double lambda_smoothness = *((const double *)(dataPtrs[3]));
            const int offset = *((int *)(dataPtrs[4]));

            double cost = 0.0;
            g.setZero();
            int num_wp = des_yaw_vec.size();
            int num_ctrl_pts = x.size() + 6;

            // Waypoint cost:
            for (int i = 0; i < num_wp; ++i)
            {
                int wp_idx = wp_idx_vec[i];
                if (wp_idx < 0 || wp_idx + 2 >= num_ctrl_pts) continue;

                int i0 = wp_idx, i1 = wp_idx + 1, i2 = wp_idx + 2;

                if (i0 >= 3 && i2 < num_ctrl_pts - 3) {
                    int i0_opt = i0 - 3;
                    int i1_opt = i1 - 3;
                    int i2_opt = i2 - 3;

                    double q1 = x[i0_opt];
                    double q2 = x[i1_opt];
                    double q3 = x[i2_opt];

                    double dq = (q1 + 4 * q2 + q3) / 6.0 - des_yaw_vec[i];
                    cost += lambda_wp * dq * dq;

                    g[i0_opt] += lambda_wp * dq * 0.33;
                    g[i1_opt] += lambda_wp * dq * 1.33;
                    g[i2_opt] += lambda_wp * dq * 0.33;
                }
            }

            // Smoothness cost:
            for (int i = 1; i < x.size() - 1; ++i)
            {
                double q_prev = x[i - 1];
                double q_curr = x[i];
                double q_next = x[i + 1];
                double d2q = q_next - 2 * q_curr + q_prev;

                cost += lambda_smoothness * d2q * d2q;
                g[i - 1] += lambda_smoothness * 2 * d2q;
                g[i]     += lambda_smoothness * -4 * d2q;
                g[i + 1] += lambda_smoothness * 2 * d2q;
            }

            return cost;
        }

        static inline NonUniformBspline optimizeYawTrajOld(const double& interval, const std::vector<double> &des_yaw_vec, const std::vector<int> &wp_idx_vec, int num_ctrl_pts) 
        {
            // Initialize control points linearly interpolating between first and last yaw values
            std::vector<double> control_points;
            control_points.resize(num_ctrl_pts);
            double yaw_start = des_yaw_vec.front();
            double yaw_end = des_yaw_vec.back();
            for (int i = 0; i < num_ctrl_pts; ++i) 
            {
                control_points[i] = yaw_start + (yaw_end - yaw_start) * (static_cast<double>(i) / (num_ctrl_pts - 1));
            }
            
            // Set up data pointers for cost function
            void *dataPtrs[4];
            double lambda_wp = 20, lambda_smoothness = 10;
            dataPtrs[0] = (void *)(&des_yaw_vec);
            dataPtrs[1] = (void *)(&wp_idx_vec);
            dataPtrs[2] = (void *)(&lambda_wp);
            dataPtrs[3] = (void *)(&lambda_smoothness);
            
            // Configure LBFGS parameters
            lbfgs::lbfgs_parameter_t lbfgs_params;
            lbfgs_params.past = 3;
            lbfgs_params.delta = 1.0e-3;
            lbfgs_params.g_epsilon = 1.0e-5;
            
            // Optimize control points using LBFGS
            double minCost;
            Eigen::VectorXd control_points_eigen = Eigen::Map<Eigen::VectorXd>(control_points.data(), control_points.size());
            int ret = lbfgs::lbfgs_optimize(control_points_eigen, 
                                minCost, 
                                cost_functional, 
                                nullptr, 
                                nullptr, 
                                dataPtrs, 
                                lbfgs_params);
            
            if (ret < 0) {
                std::cerr << "L-BFGS optimization failed with error code: " << ret << std::endl;
            }
        
            // Convert 1D yaw control points to 3D control points (y and z will be 0)
            Eigen::MatrixXd control_points_3d(control_points.size(), 1);
        
            for (int i = 0; i < control_points.size(); ++i) {
                control_points_3d(i, 0) = control_points[i]; // x coordinate is the yaw value
            }

            NonUniformBspline yaw_traj;
            yaw_traj.setUniformBspline(control_points_3d, 1, interval);
            return yaw_traj;
        }

        static inline Eigen::MatrixXd optimizeYawTrajCPOld(const double& interval, const std::vector<double> &des_yaw_vec, const std::vector<int> &wp_idx_vec, int num_ctrl_pts) 
        {
            // Initialize control points linearly interpolating between first and last yaw values
            std::vector<double> control_points;
            control_points.resize(num_ctrl_pts);
            double yaw_start = des_yaw_vec.front();
            double yaw_end = des_yaw_vec.back();
            for (int i = 0; i < num_ctrl_pts; ++i) 
            {
                control_points[i] = yaw_start + (yaw_end - yaw_start) * (static_cast<double>(i) / (num_ctrl_pts - 1));
            }
            
            // Set up data pointers for cost function
            void *dataPtrs[4];
            double lambda_wp = 20, lambda_smoothness = 10;
            dataPtrs[0] = (void *)(&des_yaw_vec);
            dataPtrs[1] = (void *)(&wp_idx_vec);
            dataPtrs[2] = (void *)(&lambda_wp);
            dataPtrs[3] = (void *)(&lambda_smoothness);
            
            // Configure LBFGS parameters
            lbfgs::lbfgs_parameter_t lbfgs_params;
            lbfgs_params.past = 3;
            lbfgs_params.delta = 1.0e-3;
            lbfgs_params.g_epsilon = 1.0e-5;
            
            // Optimize control points using LBFGS
            double minCost;
            Eigen::VectorXd control_points_eigen = Eigen::Map<Eigen::VectorXd>(control_points.data(), control_points.size());
            int ret = lbfgs::lbfgs_optimize(control_points_eigen, 
                                minCost, 
                                cost_functional, 
                                nullptr, 
                                nullptr, 
                                dataPtrs, 
                                lbfgs_params);
            
            if (ret < 0) {
                std::cerr << "L-BFGS optimization failed with error code: " << ret << std::endl;
            }
        
            // Convert 1D yaw control points to 3D control points (y and z will be 0)
            Eigen::MatrixXd control_points_3d(control_points.size(), 1);
        
            for (int i = 0; i < control_points.size(); ++i) {
                control_points_3d(i, 0) = control_points[i]; // x coordinate is the yaw value
            }
            return control_points_3d;
        }


        static inline Eigen::MatrixXd optimizeYawTrajCP(
            const double& interval,
            const std::vector<double> &des_yaw_vec,
            const std::vector<int> &wp_idx_vec,
            int num_ctrl_pts,
            const Eigen::Vector3d& fixed_start,
            const Eigen::Vector3d& fixed_end)
        {
            const int num_opt_pts = num_ctrl_pts - 6;
        
            // Build full control points array with fixed start and end
            Eigen::VectorXd full_ctrl_pts(num_ctrl_pts);
            for (int i = 0; i < 3; ++i)
                full_ctrl_pts[i] = fixed_start[i];
            for (int i = 0; i < 3; ++i)
                full_ctrl_pts[num_ctrl_pts - 3 + i] = fixed_end[i];
        
            // Linearly initialize interior control points
            double start_yaw = fixed_start[2];
            double end_yaw = fixed_end[0];
            for (int i = 0; i < num_opt_pts; ++i) {
                double ratio = static_cast<double>(i) / (num_opt_pts - 1);
                full_ctrl_pts[3 + i] = start_yaw + ratio * (end_yaw - start_yaw);
            }
        
            // Set up data pointers
            void *dataPtrs[5];
            double lambda_wp = 20, lambda_smoothness = 10;
            int offset = 3;
            dataPtrs[0] = (void *)(&des_yaw_vec);
            dataPtrs[1] = (void *)(&wp_idx_vec);
            dataPtrs[2] = (void *)(&lambda_wp);
            dataPtrs[3] = (void *)(&lambda_smoothness);
            dataPtrs[4] = (void *)(&offset);
        
            // Set up LBFGS parameters
            lbfgs::lbfgs_parameter_t lbfgs_params;
            lbfgs_params.past = 3;
            lbfgs_params.delta = 1.0e-3;
            lbfgs_params.g_epsilon = 1.0e-5;
        
            // Optimize only the interior control points
            Eigen::VectorXd x = full_ctrl_pts.segment(3, num_opt_pts);
            Eigen::VectorXd grad(num_opt_pts);
            double minCost;
        
            int ret = lbfgs::lbfgs_optimize(
                x,
                minCost,
                cost_functional_fixed,
                nullptr,
                nullptr,
                dataPtrs,
                lbfgs_params);
        
            if (ret < 0) {
                std::cerr << "L-BFGS optimization failed with code: " << ret << std::endl;
            }
        
            // Update interior control points
            full_ctrl_pts.segment(3, num_opt_pts) = x;
        
            // Convert to 1D matrix
            Eigen::MatrixXd control_points_3d(num_ctrl_pts, 1);
            for (int i = 0; i < num_ctrl_pts; ++i)
                control_points_3d(i, 0) = full_ctrl_pts[i];
            
            return control_points_3d;
        }


};