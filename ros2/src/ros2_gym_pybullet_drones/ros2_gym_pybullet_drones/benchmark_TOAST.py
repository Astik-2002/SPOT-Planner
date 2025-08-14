import subprocess, time, signal, os, rclpy
from rclpy.node import Node
from std_msgs.msg import String

num_success = 0
num_failure = 0
TIMEOUT_DURATION = 600  # 10 minutes in seconds

class MonitorNode(Node):
    def __init__(self):
        super().__init__('monitor')
        self.result = None
        self.create_subscription(String, '/simulation_status', self.cb, 1)

    def cb(self, msg):
        self.result = msg.data

def run_trial(trial):
    global num_success, num_failure

    print(f"=== Trial {trial} starting ===")

    # Start each process in its own process group
    procs = [
        subprocess.Popen(
            ["ros2", "run", "rrt_path_finder", "trajectory_server_yaw"],
            preexec_fn=os.setsid
        ),
        subprocess.Popen(
            ["ros2", "run", "rrt_path_finder", "rrt_dynamic"],
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

    print(f"Trial {trial} ended: {outcome}")

    # Terminate all child processes
    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already dead
        p.wait()  # Clean up process resources

    monitor.destroy_node()

def main():
    rclpy.init()
    try:
        for i in range(1, 21):
            run_trial(i)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping all trials early.")
    finally:
        rclpy.shutdown()
        print(f"num_failures: {num_failure}, num_success: {num_success}")

if __name__ == "__main__":
    main()