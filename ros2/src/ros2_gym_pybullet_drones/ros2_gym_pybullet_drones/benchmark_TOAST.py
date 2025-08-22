import subprocess, time, signal, os, rclpy
from rclpy.node import Node
from std_msgs.msg import String

num_success = 0
num_failure = 0
num_crash_failures = 0
num_process_failures = 0
num_timeout_failures = 0
num_unknown_failures = 0

TIMEOUT_DURATION = 300  # 10 minutes in seconds

class MonitorNode(Node):
    def __init__(self):
        super().__init__('monitor')
        self.result = None
        self.create_subscription(String, '/simulation_status', self.cb, 1)

    def cb(self, msg):
        self.result = msg.data

def run_trial(trial):
    global num_success, num_failure
    global num_crash_failures, num_process_failures, num_timeout_failures, num_unknown_failures

    print(f"=== Trial {trial} starting ===")

    # Start each process in its own process group
    procs = [
        subprocess.Popen(
            ["ros2", "run", "rrt_path_finder", "trajectory_server_yaw"],
            preexec_fn=os.setsid
        ),
        subprocess.Popen(
            ["ros2", "run", "rrt_path_finder", "pcd_manipulation"],
            preexec_fn=os.setsid
        ),
        subprocess.Popen(
            ["ros2", "run", "rrt_path_finder", "rrt_dynamic"],
            preexec_fn=os.setsid
        ),
        # subprocess.Popen(
        #     ["ros2", "run", "rrt_path_finder", "new_replan"],
        #     preexec_fn=os.setsid
        # ),
        subprocess.Popen(
            ["ros2", "run", "ros2_gym_pybullet_drones", "model_publisher"],
            preexec_fn=os.setsid
        ),
        subprocess.Popen(
            ["ros2", "run", "ros2_gym_pybullet_drones", "trajectory_tracker"],
            preexec_fn=os.setsid
        )
    ]

    monitor = MonitorNode()
    start_time = time.time()
    outcome = None
    timeout_occurred = False

    while rclpy.ok() and outcome is None:
        # Check for timeout
        if time.time() - start_time > TIMEOUT_DURATION:
            timeout_occurred = True
            outcome = "timeout"
            break

        try:
            rclpy.spin_once(monitor, timeout_sec=0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Stopping processes...")
            for p in procs:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            raise  # Let the main loop handle the exit

        # Check if any child process terminated unexpectedly
        for p in procs:
            if p.poll() is not None:  # Process terminated
                outcome = "process_crash"
                break

        # Check if simulation status received
        if monitor.result is not None:
            # Could be "goal_reached", "uav_crash", etc.
            outcome = monitor.result

    # Determine outcome if not set by above checks
    if outcome is None:
        if timeout_occurred:
            outcome = "timeout"
        else:
            outcome = "unknown_error"

    # Update success/failure counts
    if outcome == "goal_reached":
        num_success += 1
    else:
        num_failure += 1
        if outcome == "crash":
            num_crash_failures += 1
        elif outcome == "process_crash":
            num_process_failures += 1
        elif outcome == "timeout":
            num_timeout_failures += 1
        else:
            num_unknown_failures += 1

    print(f"Trial {trial} ended: {outcome}")

    # Terminate all child processes
    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already dead
        p.wait()  # Clean up process resources

    monitor.destroy_node()
    if outcome == "process_crash":
        return False
    else:
        return True
    

def main():
    rclpy.init()
    try:
        for i in range(1, 51):
            a = run_trial(i)
            # if a == False:
            #     print("process crash debugging")
            #     break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping all trials early.")
    finally:
        rclpy.shutdown()
        print(f"Summary:")
        print(f"  num_success: {num_success}")
        print(f"  num_failures: {num_failure}")
        print(f"    - UAV crashes: {num_crash_failures}")
        print(f"    - Process crashes: {num_process_failures}")
        print(f"    - Timeouts: {num_timeout_failures}")
        print(f"    - Unknown errors: {num_unknown_failures}")

if __name__ == "__main__":
    main()
