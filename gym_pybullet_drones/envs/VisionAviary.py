import os
import numpy as np
from gym import spaces
import pkg_resources
import pybullet as p
import open3d as o3d
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from scipy.spatial.transform import Rotation as R
import random
import csv

class VisionAviary(BaseAviary):
    """Multi-drone environment class for control applications using vision."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 environment_file = None,
                 dynamic_obs = False,
                 static_obs = False,
                 num_obstacles = 0,
                 realistic = False,
                 torus = False,
                 deg360 = False
                 ):
        """Initialization of an aviary environment for control applications using vision.

        Attribute `vision_attributes` is automatically set to True when calling
        the superclass `__init__()` method.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """

        self.environment_file = environment_file
        self.is_dyn = dynamic_obs
        self.static = static_obs
        self.obs = obstacles
        self.dynamic_obstacles = []
        self.near = 0.1
        self.far = 1000
        self.num_obstacles = num_obstacles
        self.realistic = realistic
        self.is_torus = torus
        self._360view = deg360
        self.obstacle_ids = []
        self.init_drone_pos = initial_xyzs
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         vision_attributes=True,
                         output_folder=output_folder
                         )
        self.VID_WIDTH=int(640)
        self.VID_HEIGHT=int(480)
        
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=90.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )

    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([0.,           0.,           0.,           0.])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Dict({str(i): spaces.Box(low=act_lower_bound,
                                               high=act_upper_bound,
                                               dtype=np.float32
                                               ) for i in range(self.NUM_DRONES)})
    
    ################################################################################
    
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
                                                                     high=obs_upper_bound,
                                                                     dtype=np.float32
                                                                     ),
                                                 "neighbors": spaces.MultiBinary(self.NUM_DRONES),
                                                 "rgb": spaces.Box(low=0,
                                                                   high=255,
                                                                   shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                                                                   dtype=np.uint8
                                                                   ),
                                                 "dep": spaces.Box(low=.01,
                                                                   high=1000.,
                                                                   shape=(self.IMG_RES[1],
                                                                    self.IMG_RES[0]),
                                                                   dtype=np.float32
                                                                   ),
                                                 "seg": spaces.Box(low=0,
                                                                   high=100,
                                                                   shape=(self.IMG_RES[1],
                                                                   self.IMG_RES[0]),
                                                                   dtype=int
                                                                   )
                                                 }) for i in range(self.NUM_DRONES)})
    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix,
        "rgb", "dep", and "seg" are matrices containing POV camera captures.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        adjacency_mat = self._getAdjacencyMatrix()
        obs = {}
        for i in range(self.NUM_DRONES):        
            self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
            #### Printing observation to PNG frames example ############
            if self.RECORD:
                self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                img_input=self.rgb[i],
                                path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                )
            obs[str(i)] = {"state": self._getDroneStateVector(i), \
                        "neighbors": adjacency_mat[i,:], \
                        "rgb": self.rgb[i], \
                        "dep": self.dep[i], \
                        "seg": self.seg[i] \
                        }
        return obs


    ################################################################################

    def simulateLidar(self):
        pcd_array = []
        for i in range(self.NUM_DRONES):
            pcd = self._simulateLivoxMid360Lidar(i)
            pcd_array.append(pcd)
        return pcd_array
                
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        clipped_action = np.zeros((self.NUM_DRONES, 4))
        for k, v in action.items():
            clipped_action[int(k), :] = np.clip(np.array(v), 0, self.MAX_RPM)
        return clipped_action

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def step(self,
             action):
        """Overrides the step method to include dynamic obstacle updates."""
        if self.is_dyn:
            self._updateDynamicObstacles()  # Update dynamic obstacle positions
        return super().step(action)

    ################################################################################

    def _addObstacles(self):
        """Add a 'forest' of trees as obstacles to the environment.
        
        Parameters
        ----------
        num_trees : int, optional
            The number of trees to add to the environment.
        x_bounds : tuple, optional
            The x-axis bounds within which trees will be randomly placed.
        y_bounds : tuple, optional
            The y-axis bounds within which trees will be randomly placed.
        """
        # Call the parent class _addObstacles (if needed)
        # super()._addObstacles()
        if self.is_dyn:
            self._addDynamicObstacles(num_obstacles=self.num_obstacles)
        if self.static:
            less = False
            if not less:
                num_trees= 80
                x_bounds=(0.5, 55.5)
                y_bounds=(-7.0, 7.0)
                
                base_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets')
                tree_urdf = os.path.join(base_path, "simple_tree.urdf")
                realistic_tree_urdfs = [os.path.join(base_path, "realistic_tree1.urdf"), os.path.join(base_path, "realistic_tree2.urdf"), os.path.join(base_path, "realistic_tree3.urdf"),
                                        os.path.join(base_path, "realistic_tree4.urdf"), os.path.join(base_path, "realistic_tree5.urdf"), os.path.join(base_path, "realistic_tree6.urdf")]
                np.random.seed(42)
                output_dir = "/home/astik/gym-pybullet-drones/gym_pybullet_drones/envs/obstacle_pos"
                if self.environment_file:
                    csv_file = os.path.join(output_dir, self.environment_file)
                    try:
                        with open(csv_file, newline="") as f:
                            reader = csv.reader(f)
                            # Skip header row if present
                            next(reader, None)
                            print("[vision aviary debug] loaded environment: ",csv_file)
                            for row in reader:
                                if len(row) < 4:
                                    continue
                                tree_id = row[0]
                                x_pos = float(row[1])
                                y_pos = float(row[2])
                                z_pos = float(row[3])
                                pos = (x_pos, y_pos, z_pos)
                                # print(f"Tree {tree_id} at position {pos}")
                                if self.realistic:
                                    tree_urdf = realistic_tree_urdfs[random.randint(0, 5)]
                                if os.path.exists(tree_urdf):
                                    p.loadURDF(tree_urdf,
                                            pos,
                                            p.getQuaternionFromEuler([0, 0, 0]),  # No rotation
                                            useFixedBase=True,
                                            physicsClientId=self.CLIENT)
                                else:
                                    print(f"File not found: {tree_urdf}")
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
            
            else:
                base_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets')
                tree_urdf = os.path.join(base_path, "simple_tree.urdf")
                pos = (0.5, 0, 0)
                if os.path.exists(tree_urdf):
                        tree_id = p.loadURDF(tree_urdf,
                                pos,
                                p.getQuaternionFromEuler([0, 0, 0]),  # No rotation
                                useFixedBase=True,
                                physicsClientId=self.CLIENT)
                        self.obstacle_ids.append(tree_id)

                else:
                    print(f"File not found: {tree_urdf}")


    def _addDynamicObstacles(self, num_obstacles=1, x_bounds=(2, 18), y_bounds=(-8, 8), velocity_range=(-0.3, 0.3)):
        """Adds dynamic obstacles that move with velocity profiles."""
        base_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets')

        obstacle_urdf = os.path.join(base_path, "red_cylinder.urdf")
        torus_urdf = os.path.join(base_path, "torus.urdf")
        radii_array = ["red_cylinder_0.2.urdf", "red_cylinder_0.3.urdf", "red_cylinder_0.4.urdf", "red_cylinder_0.5.urdf"]
        for i in range(num_obstacles):
            num = random.randrange(4)
            obstacle_urdf = os.path.join(base_path, radii_array[num])
            if(num == 0):
                radius = 0.2
            if(num == 1):
                radius = 0.3
            if(num == 2):
                radius = 0.4
            if(num == 3):
                radius = 0.5
            # Generate random initial positions within bounds
            d_uav = -1
            while d_uav < 2.0:
                x_pos = np.random.uniform(x_bounds[0], x_bounds[1])
                y_pos = np.random.uniform(y_bounds[0], y_bounds[1])
                z_pos = 0.5  # Fixed height for the obstacles
                pos = (x_pos, y_pos, z_pos)
                pos_obs = np.array(pos)
                d_uav = np.linalg.norm(self.init_drone_pos - pos_obs)

            # Generate random velocity components
            velocity = [
            np.random.uniform(velocity_range[0], velocity_range[1]),
            np.random.uniform(velocity_range[0], velocity_range[1]),
            0.0  # Obstacles move in the XY plane only
            ]

            # Decide URDF based on is_torus and 30% probability
            type_obs = 0
            if self.is_torus and np.random.rand() < 0.3:
                urdf_to_use = torus_urdf
                type_obs = 1
            else:
                urdf_to_use = obstacle_urdf

            if os.path.exists(urdf_to_use):
                obstacle_id = p.loadURDF(urdf_to_use,
                         pos,
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=False,
                         physicsClientId=self.CLIENT)
                self.dynamic_obstacles.append({
                    "pos": pos,
                    "id": obstacle_id,
                    "velocity": velocity,
                    "radius":radius,
                    "type":type_obs
                })
                self.obstacle_ids.append(obstacle_id)
            else:
                print(f"File not found: {urdf_to_use}")

    def _updateDynamicObstacles(self, range_x = [2, 18], range_y = [-8, 8]):
        """Updates the positions of dynamic obstacles based on their velocities."""
        for obstacle in self.dynamic_obstacles:
            obstacle_id = obstacle["id"]
            velocity = obstacle["velocity"]

            # Get the current position of the obstacle
            pos, _ = p.getBasePositionAndOrientation(obstacle_id, physicsClientId=self.CLIENT)
            vx, vy, vz = obstacle["velocity"]
            
            # Calculate proposed new position
            new_pos = [pos[0] + vx / self.SIM_FREQ,
                    pos[1] + vy / self.SIM_FREQ,
                    pos[2] + vz / self.SIM_FREQ]
            
            # Define your boundaries (adjust these values as needed)
            x_min, x_max = range_x[0], range_x[1]
            y_min, y_max = range_y[0], range_y[1]
            
            # Check for boundary collisions and handle bouncing
            bounced = False
            
            # X-axis boundary check
            if new_pos[0] < x_min:
                new_pos[0] = x_min + (x_min - new_pos[0])  # Reflect the overshoot
                vx = -vx
                bounced = True
            elif new_pos[0] > x_max:
                new_pos[0] = x_max - (new_pos[0] - x_max)  # Reflect the overshoot
                vx = -vx
                bounced = True
                
            # Y-axis boundary check
            if new_pos[1] < y_min:
                new_pos[1] = y_min + (y_min - new_pos[1])  # Reflect the overshoot
                vy = -vy
                bounced = True
            elif new_pos[1] > y_max:
                new_pos[1] = y_max - (new_pos[1] - y_max)  # Reflect the overshoot
                vy = -vy
                bounced = True
                
            # Update velocity if bounced
            if bounced:
                obstacle["velocity"] = [vx, vy, vz]
            
            # Set the new position of the obstacle
            p.resetBasePositionAndOrientation(obstacle_id,
                                            new_pos,
                                            p.getQuaternionFromEuler([0, 0, 0]),
                                            physicsClientId=self.CLIENT)
            obstacle["pos"] = new_pos
    
    def _checkCollision(self, drone_id):
        """Check if drone collides with obstacles or ground."""
        # Get drone position and closest obstacle
        drone_pos = self.pos[drone_id]
        drone_body_id = self.DRONE_IDS[drone_id]
        
        # Check ground collision (z-coordinate)
        if drone_pos[2] < 0.1:
            return True
        
        # Check obstacle collisions
        contact_points = p.getContactPoints(
            bodyA=drone_body_id,
            physicsClientId=self.CLIENT
        )
        for contact in contact_points:
            if contact[2] == -1 or contact[2] in self.obstacle_ids:
                return True
        return False
    # def _addObstacles(self):
    #     """Add obstacles to the environment, including multiple cylinders of different colors at fixed positions."""
    #     super()._addObstacles()
    #     base_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets')
    #     cylinder_colors = ['red']
    #     cylinders = [os.path.join(base_path, f"{color}_cylinder.urdf") for color in cylinder_colors for _ in range(9)]
    #     # obstacles = os.path.join(base_path, "gate.urdf")
    #     # Fixed positions
    #     self.fixed_positions = [
    #         (1.0, 1.0, 0.5),
    #         (1.0, 0, 0.5),
    #         (1.0, -1.0, 0.5),
    #         (0, 1.0, 0.5),
    #         (0, 5, 0.5),
    #         (0, -1.0, 0.5),
    #         (-1.0, 1.0, 0.5),
    #         (-1.0, 0, 0.5),
    #         (-1.0, -1.0, 0.5)
    #     ]

    #     # obstacle_pos = (0.0, 0.0, 0.0)

    #     for urdf, pos in zip(cylinders, self.fixed_positions):
    #         if os.path.exists(urdf):
    #             p.loadURDF(urdf,
    #                     pos,
    #                     p.getQuaternionFromEuler([0, 0, 0]),
    #                     useFixedBase=True,
    #                     physicsClientId=self.CLIENT
    #                     )
    #         else:
    #             print(f"File not found: {urdf}")

    #     # if os.path.exists(obstacles):
    #     #     p.loadURDF(obstacles,
    #     #                 obstacle_pos,
    #     #                 p.getQuaternionFromEuler([0, 0, 0]),
    #     #                 useFixedBase=True,
    #     #                 physicsClientId=self.CLIENT)
    #     # else:
    #     #     print(f"File not found: {obstacles}")



