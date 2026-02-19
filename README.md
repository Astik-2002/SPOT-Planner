# IMPORTANT NOTE:
This repository provides an implementation of SPOT algorithm in gym pybullet environment. 

```bash
$ conda create -n drones python=3.8
$ conda activate drones
$ pip3 install --upgrade pip
$ git clone https://github.com/Astik-2002/SPOT-Spatio-Temporal-Obstacle-free-Trajectory-Planning-for-UAVs-in-unknown-dynamic-environments
$ cd SPOT-Spatio-Temporal-Obstacle-free-Trajectory-Planning-for-UAVs-in-unknown-dynamic-environments/
$ pip3 install -e .
```
<!--
On Ubuntu and with a GPU available, optionally uncomment [line 203](https://github.com/utiasDSL/gym-pybullet-drones/blob/fab619b119e7deb6079a292a04be04d37249d08c/gym_pybullet_drones/envs/BaseAviary.py#L203) of `BaseAviary.py` to use the [`eglPlugin`](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.778da594xyte)
-->

## Examples

The core algorithm is implemented in rrt_path_finder ros2 package in [`SPOT-Spatio-Temporal-Obstacle-free-Trajectory-Planning-for-UAVs-in-unknown-dynamic-environments/ros2/`]
Directory super_planner provides the implementation of SUPER (https://github.com/hku-mars/SUPER) in the gym pybullet environment. For running the planner, run the command

```bash
$ ros2 run ros2_gym_pybullet_drones benchmark_toast
```
