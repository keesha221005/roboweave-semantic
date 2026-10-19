import json
import threading
import time
import atexit
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SCADAROSEngine(Node):
    def __init__(self, state_ref):
        super().__init__('scada_master_engine')
        self.state = state_ref
        
        # Publishers and Subscriptions
        self.pub_dispatch = self.create_publisher(String, '/warehouse/mission_dispatch', 10)
        self.pub_mesh = self.create_publisher(String, '/warehouse/p2p_mesh', 10)
        
        self.create_subscription(String, '/warehouse/telemetry', self.telem_cb, 50)
        self.create_subscription(String, '/warehouse/p2p_mesh', self.mesh_cb, 50)

        # Dynamic Obstacle Tracking for UI Injections
        self.active_obstacles = {}
        
        # WMS Task Queue Tracking
        self.task_queue = []
        self.completed_tasks_count = 0
        
        # Background timer to heartbeat the phantom obstacle tokens to the mesh (0.5s interval)
        self.obs_timer = self.create_timer(0.5, self.broadcast_obstacles)

    def broadcast_obstacles(self):
        try:
            for obs_id, pos in self.active_obstacles.items():
                phantom_payload = {
                    "sender_id": obs_id,
                    "priority": 99,  # Absolute highest priority forces AMRs to route around
                    "status": "BLOCKED",
                    "pos": pos,
                    "reservations": [[pos[0], pos[1], 0]], 
                    "time_step": 0
                }
                self.pub_mesh.publish(String(data=json.dumps(phantom_payload)))
        except Exception:
            pass

    def inject_obstacle(self, r, c):
        obs_id = f"OBS_{r}_{c}"
        with self.state["lock"]:
            self.active_obstacles[obs_id] = [r, c]
            self.state["mesh_logs"].append(f"[OBSTACLE INJECTED] Dynamic blockage at grid [{r}, {c}]")
            if len(self.state["mesh_logs"]) > 6:
                self.state["mesh_logs"].pop(0)

    def clear_obstacles(self):
        with self.state["lock"]:
            self.active_obstacles.clear()
            self.state["mesh_logs"].append("[OBSTACLES CLEARED] Mesh returned to nominal state.")
            if len(self.state["mesh_logs"]) > 6:
                self.state["mesh_logs"].pop(0)

    def add_task_to_queue(self, goal, priority, item_desc):
        with self.state["lock"]:
            task = {"goal": goal, "priority": priority, "item": item_desc, "status": "QUEUED"}
            self.task_queue.append(task)
            self.state["mesh_logs"].append(f"[WMS QUEUE] Task added for goal {goal} ({item_desc})")
            if len(self.state["mesh_logs"]) > 6:
                self.state["mesh_logs"].pop(0)
        
        # Call dispatch outside the primary state lock context to prevent thread starvation
        self.try_dispatch_next_task()

    def try_dispatch_next_task(self):
        best_robot = None
        target_goal = None
        priority = 2
        item = "Cargo"
        min_distance = float('inf')

        # Scope the lock strictly to data evaluation
        with self.state["lock"]:
            if not self.task_queue:
                return
            
            next_task = self.task_queue[0]
            target_goal = next_task["goal"]
            priority = next_task["priority"]
            item = next_task["item"]

            for r_id, amr in self.state["fleet"].items():
                if amr["status"] in ["COMPLETED", "STANDBY"]:
                    curr_pos = amr["pos"]
                    dist = abs(curr_pos[0] - target_goal[0]) + abs(curr_pos[1] - target_goal[1])
                    if dist < min_distance:
                        min_distance = dist
                        best_robot = r_id
                    
            if best_robot and self.task_queue:
                next_task = self.task_queue.pop(0)
                next_task["status"] = "ASSIGNED"
                self.state["mesh_logs"].append(f"[DYNAMIC WMS] Nearest AMR {best_robot} assigned (dist: {min_distance} cells)")
                if len(self.state["mesh_logs"]) > 6:
                    self.state["mesh_logs"].pop(0)

        # Publish outside the lock
        if best_robot and target_goal:
            payload = {
                "target_robot": best_robot,
                "goal": target_goal,
                "priority": priority,
                "item": item
            }
            self.pub_dispatch.publish(String(data=json.dumps(payload)))

    def telem_cb(self, msg):
        try:
            data = json.loads(msg.data)
            r_id = data["id"]
            new_pos = tuple(data["pos"])
            status = data["status"]
            battery = float(data["battery"])
            path = [tuple(p) for p in data.get("path", [])]
            item = data.get("item", "Cargo Unit")

            with self.state["lock"]:
                if r_id not in self.state["fleet"]:
                    self.state["fleet"][r_id] = {
                        "name": r_id, "start": new_pos, "goal": tuple(data["goal"]),
                        "src_name": "BAY_IN", "dst_name": "STATION_BAY", "item": item,
                        "priority": data["priority"], "status": status, "pos": new_pos,
                        "battery": battery, "path": path, "trail": [new_pos],
                        "ticks": {"cargo": 1, "empty": 0, "docking": 0, "blocked": 0}
                    }
                
                amr = self.state["fleet"][r_id]
                if amr["trail"] and amr["trail"][-1] != new_pos:
                    amr["trail"].append(new_pos)
                
                amr["pos"] = new_pos
                amr["goal"] = tuple(data["goal"])
                amr["status"] = status
                amr["battery"] = battery
                amr["path"] = path
                amr["item"] = item
                
                amr["ticks"]["cargo" if status == "NAVIGATING" else ("docking" if status == "COMPLETED" else "blocked")] += 1

                if status == "COMPLETED" and amr.get("logged_goal") != new_pos:
                    amr["logged_goal"] = new_pos
                    self.state["transport_history"].insert(0, {
                        "Robot": r_id, "Timestamp": time.strftime("%H:%M:%S"),
                        "Source": f"{amr['start']}", "Destination": f"{amr['goal']}",
                        "Item": item, "Ride Time": f"{len(amr['trail']) * 0.5:.1f}s",
                        "Status": "DELIVERED (ROS 2)"
                    })
                    # Automatically trigger the next optimal task assignment from the queue
                    self.try_dispatch_next_task()
        except Exception:
            pass

    def mesh_cb(self, msg):
        try:
            data = json.loads(msg.data)
            sender_id = data["sender_id"]
            if sender_id.startswith("OBS_"):
                return
                
            with self.state["lock"]:
                self.state["mesh_logs"].append(f"[{sender_id}] P{data['priority']} Mesh Token: {data.get('reservations', [])[:2]}")
                if len(self.state["mesh_logs"]) > 6:
                    self.state["mesh_logs"].pop(0)
        except Exception:
            pass

    def dispatch_mission(self, target_robot, goal, priority, item_desc):
        payload = {
            "target_robot": target_robot,
            "goal": goal,
            "priority": priority,
            "item": item_desc
        }
        self.pub_dispatch.publish(String(data=json.dumps(payload)))

def init_ros_engine():
    if not rclpy.ok():
        rclpy.init()
        
    shared_state = {
        "lock": threading.Lock(),
        "fleet": {},
        "mesh_logs": [],
        "transport_history": [],
        "action_point_stats": {"INBOUND_DOCK": 12, "OUTBOUND_BAY": 24, "CENTRAL_CORRIDOR": 18}
    }
    
    node = SCADAROSEngine(shared_state)
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()

    def cleanup():
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    atexit.register(cleanup)
    return node, shared_state