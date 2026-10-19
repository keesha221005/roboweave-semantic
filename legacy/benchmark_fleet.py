import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

class BenchmarkNode(Node):
    def __init__(self):
        super().__init__('benchmark_node')
        # Subscribing to telemetry to track live state changes
        self.sub_telemetry = self.create_subscription(String, '/warehouse/telemetry', self.telemetry_cb, 10)
        
        self.task_starts = {}
        self.task_latencies = []
        self.completed_tasks = 0
        self.deadlock_recoveries = 0
        
        self.previous_states = {}

        self.get_logger().info("Roboweave Benchmark Node Started. Listening for empirical data...")

    def telemetry_cb(self, msg):
        try:
            data = json.loads(msg.data)
            robot_id = data['id']
            status = data['status']
            current_time = time.time()

            prev_status = self.previous_states.get(robot_id, "STANDBY")

            # 1. Detect Dispatch (Start Timer)
            if status == "NAVIGATING" and prev_status != "NAVIGATING":
                self.task_starts[robot_id] = current_time
                self.get_logger().info(f"[TIMER STARTED] {robot_id} received mission.")

            # 2. Detect Autonomous Deadlock Resolution Events
            if status in ["BLOCKED", "YIELDING"] and prev_status not in ["BLOCKED", "YIELDING"]:
                self.deadlock_recoveries += 1
                self.get_logger().warn(f"[DEADLOCK DETECTED] {robot_id} entered {status}. Total fleet conflict events: {self.deadlock_recoveries}")

            # 3. Detect Completion (End Timer & Calculate)
            if status == "GOAL_REACHED" and prev_status != "GOAL_REACHED":
                if robot_id in self.task_starts:
                    latency = current_time - self.task_starts[robot_id]
                    self.task_latencies.append(latency)
                    self.completed_tasks += 1
                    
                    avg_latency = sum(self.task_latencies) / len(self.task_latencies)
                    
                    self.get_logger().info(f"[MISSION COMPLETE] {robot_id} finished in {latency:.2f}s")
                    self.get_logger().info(f"--- LIVE METRICS | Completed: {self.completed_tasks} | Avg Latency: {avg_latency:.2f}s | Deadlocks Encountered: {self.deadlock_recoveries} ---")
                    
                    # Clear the start time so it is ready for the next queue dispatch
                    del self.task_starts[robot_id]

            self.previous_states[robot_id] = status

        except Exception as e:
            self.get_logger().debug(f"Telemetry parse error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Print final auditable benchmark report on terminal exit
        print("\n==================================================")
        print(" FINAL ROBOWEAVE BENCHMARK REPORT ")
        print("==================================================")
        print(f" Total Tasks Completed : {node.completed_tasks}")
        if node.task_latencies:
            print(f" Fleet Makespan (Max)  : {max(node.task_latencies):.2f} seconds")
            print(f" Average Task Latency  : {sum(node.task_latencies)/len(node.task_latencies):.2f} seconds")
        print(f" Deadlocks Autonomously Resolved : {node.deadlock_recoveries}")
        print("==================================================\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()