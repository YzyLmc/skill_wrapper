import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped
from pose_estimation_msgs.srv import EstimatePose
from tf2_ros import TransformBroadcaster, TransformListener, Buffer, ExtrapolationException, ConnectivityException, LookupException
from rclpy.duration import Duration
import math
import time
import sys
from scipy.spatial.transform import Rotation as R

from panda_utils.panda_arm import ArmControl
from panda_utils.configs import PICKPLACE_JOINT_HOME, PICKPLACE_EE_HOME
import tf_transformations as tft

class Executor(Node):
    def __init__(self):
        super().__init__('executor')


        self.WIPE_DEMO_PATH="raw_datasets/ranch_final_2025-05-15-19-37/episode_0000.npy"
        self.STACK_DEMO_PATH="raw_datasets/stack_final_2025-05-21-17-52/episode_0000.npy"
        self.TILT_TEAPOT_DEMO_PATH="raw_datasets/tilt_teapot_2025-05-15-22-55/episode_0000.npy"
        self.PICK_SPONGE_DEMO_PATH="raw_datasets/sponge_final_2025-05-22-17-38/episode_0000.npy"
        self.RAISE_DEMO_PATH="raw_datasets/raise_final_2025-05-22-17-48/episode_0000.npy"
        self.PLACE_SPONGE_DEMO_PATH="raw_datasets/place_sponge_final_2025-05-23-21-49/episode_0000.npy"
        self.STACK_BOWL_DEMO_PATH_1="raw_datasets/stack_red_bowl_1_2025-06-22-22-51/episode_0000.npy"
        self.STACK_BOWL_DEMO_PATH_2="raw_datasets/stack_red_bowl_2_2025-06-22-22-55/episode_0000.npy"
        self.STACK_BOWL_DEMO_PATH_1_TEST="raw_datasets/stack_red_bowl_1_test_2025-06-25-08-59/episode_0000.npy"
        self.STACK_BOWL_DEMO_PATH_2_TEST="raw_datasets/stack_red_bowl_2_test_2025-06-25-09-03/episode_0000.npy"

        self.bridge = CvBridge()
        self.client = self.create_client(EstimatePose, '/detect_object_pose')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot = ArmControl(joint_home=PICKPLACE_JOINT_HOME)

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_info = None
        self.request_sent = False
        self.transform_ready = False
        self.latest_transform = None
        self.picked_obj = None
        self.picked_quat = None

        self.obj2pose = {}
        #NOTE: only for the absolute offset from foundation pose
        self.obj2offset = {
            #change offset everytime
            "cyan_cube": np.array([-0.01, 0.01, 0.00]),
            "expo_spray": np.array([0.0, 0.0, 0.13]),
            "purple_teapot": np.array([-0.02, -0.06, 0.1]),
            "lemon": np.array([0.19, 0.07, -0.25]),
            "white_mug": np.array([0.65, 0.87, 0.0]), # for handle np.array([0.95, 0.7, -0.1])
            "spoon" : np.array([0.01, 0.00, 0.065]),
            "sponge": np.array([0.25, 0.52, -0.15]),
            "bowl": np.array([0.03, 0.07, 0.04])
        }
        #NOTE: separate offset for grasping
        self.obj2offset_picking = {
            "purple_teapot": None
        }

        self.obj2pickparams = {
            "cyan_cube":{
                "close_force": 50.0, #default
                "close_width": 0.0, #default
            },
            "lemon": {
                "close_force": 5.0,
                "close_width": 0.07
            },
            "white_mug":{
                "close_force": 50.0, #default
                "close_width": 0.0, #default
            },
            "spoon":{
                "close_force": 50.0, #default
                "close_width": 0.0, #default
            },
            "purple_teapot":{
                "close_force": 50.0, #default 
                "close_width": 0.00, #default
            },
            "sponge":{
                "close_force": 20.0, 
                "close_width": 0.05,
            },
            "bowl":{
                "close_force": 50.0, #default
                "close_width": 0.00, #default
            }
        }

        self.create_subscription(Image, '/dave/dave_camera/color/image_raw', self._rgb_callback, 10)
        self.create_subscription(Image, '/dave/dave_camera/aligned_depth_to_color/image_raw', self._depth_callback, 10)
        self.create_subscription(CameraInfo, '/dave/dave_camera/color/camera_info', self._info_callback, 10)

        # # mug's base pose at collection time
        # self.mug_base_pose = np.array([
        # [-0.71240137, -0.01749061, -0.70155425,  0.64854837],
        # [-0.70092025,  0.06697977,  0.71008768, -0.02775536],
        # [ 0.03457008,  0.99760102, -0.059976  , -0.01596638],
        # [ 0.        ,  0.        ,  0.        ,  1.        ]]) 
        

        # # saved “base → white_mug” transform
        # self.saved_tf = TransformStamped()
        # self.saved_tf.header.frame_id = 'base'
        # self.saved_tf.child_frame_id = 'white_mug'
        # self.saved_tf.transform.translation.x = 0.6292856882215441
        # self.saved_tf.transform.translation.y = -0.033799726848773595
        # self.saved_tf.transform.translation.z = 0.13515665807463878
        # self.saved_tf.transform.rotation.x = 0.01281473815596506
        # self.saved_tf.transform.rotation.y = 0.7362418951518231
        # self.saved_tf.transform.rotation.z = 0.6723021073950988
        # self.saved_tf.transform.rotation.w = 0.07611524618205433

        # self.M_base_mug = self._tf_to_mat(self.saved_tf)
        # self.M_mug_base = np.linalg.inv(self.M_base_mug)


        # self.robot.close_gripper(width=0.02, force=5.0) #play around with this

    def go_home(self):
        quat = np.array([PICKPLACE_EE_HOME['qx'], PICKPLACE_EE_HOME['qy'], PICKPLACE_EE_HOME['qz'], PICKPLACE_EE_HOME['qw']])

        if self.picked_obj == "purple_teapot" or self.picked_obj == "bowl":
            # Convert quaternions to rotation objects
            rotations = R.from_quat(quat)
            
            # Convert rotation objects to Euler angles (roll, pitch, yaw)
            rpy_array = rotations.as_euler('xyz', degrees=True)
            if self.picked_obj == "bowl":
                rpy_array[2]+=90
            else:
                rpy_array[2]-=90

            rotations = R.from_euler('xyz', rpy_array, degrees=True)
            quat = rotations.as_quat()

        self.robot.goto(PICKPLACE_EE_HOME['x'], PICKPLACE_EE_HOME['y'], PICKPLACE_EE_HOME['z'], quat)


    def close_gripper(self, width=0.00, force=50.0):
        self.robot.close_gripper(width=width, force=force)

    def open_gripper(self):
        self.robot.open_gripper()

    def _tf_to_mat(self, tf_st):
        q = (
            tf_st.transform.rotation.x,
            tf_st.transform.rotation.y,
            tf_st.transform.rotation.z,
            tf_st.transform.rotation.w,
        )
        M = tft.quaternion_matrix(q)
        M[0:3,3] = (
            tf_st.transform.translation.x,
            tf_st.transform.translation.y,
            tf_st.transform.translation.z,
        )
        return M


    def _pose_to_mat(self, p):
        q = (p[3], p[4], p[5], p[6])
        M = tft.quaternion_matrix(q)
        M[0:3,3] = (p[0], p[1], p[2])
        return M


    def _mat_to_pose(self, m):
        x, y, z = m[0:3,3]
        q = tft.quaternion_from_matrix(m)
        qx, qy, qz, qw = q
        return np.array([x,y,z,qx,qy,qz,qw])


    def _rgb_callback(self, msg):
        self.latest_rgb = msg


    def _depth_callback(self, msg):
        self.latest_depth = msg


    def _info_callback(self, msg):
        self.latest_info = msg


    def _request_pose(self, obj):
        if not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Pose estimation service not available.")
            return
    
        if self.latest_rgb is None or self.latest_depth is None or self.latest_info is None:
            self.get_logger().warn("Sensor data not ready. Skipping pose request.")
            return

        self.get_logger().info(f"[REQUEST] Sending pose estimation for: {obj}")

        req = EstimatePose.Request()
        req.object_name = obj
        req.rgb = self.bridge.cv2_to_imgmsg(
            self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding="rgb8"), encoding="rgb8"
        )
        req.depth = self.bridge.cv2_to_imgmsg(
            self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="16UC1").astype(np.float32) / 1000.0,
            encoding="32FC1"
        )
        req.info = self.latest_info

        future = self.client.call_async(req)
        future.add_done_callback(lambda f: self._pose_callback(f, obj))
        self.request_sent = True


    def _pose_callback(self, future, obj):
        try:
            response = future.result()
            self.get_logger().info(f"[RESPONSE] for {obj}: object_found={response.object_found}, confidence={response.confidence:.3f}")

            if not response.object_found:
                self.get_logger().warn("Object not found.")
                return

            self.latest_transform = TransformStamped()
            self.latest_transform.header.stamp = self.get_clock().now().to_msg()

            self.latest_transform.header.frame_id = "dave_color_optical_frame"

            self.latest_transform.child_frame_id = obj
            self.latest_transform.transform.translation.x = response.pose.pose.position.x
            self.latest_transform.transform.translation.y = response.pose.pose.position.y
            self.latest_transform.transform.translation.z = response.pose.pose.position.z
            self.latest_transform.transform.rotation = response.pose.pose.orientation

            for _ in range(10):
                self.tf_broadcaster.sendTransform(self.latest_transform)
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.05)

            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info(f"Published TF for {obj}")
            self.transform_ready = True

        except Exception as e:
            self.get_logger().error(f"Pose service call failed for {obj}: {e}")


    def transform_to_matrix(self, tf):
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        trans = tf.transform.translation
        rot = tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        T[:3, 3] = [trans.x, trans.y, trans.z]
        return T


    def get_pose(self, obj):
        self.request_sent = False
        self.transform_ready = False
        self._request_pose(obj)

        while not self.transform_ready:
            rclpy.spin_once(self, timeout_sec=0.2)

        while not self.tf_buffer.can_transform(
            "dave_color_optical_frame", obj, rclpy.time.Time()
        ):
            self.get_logger().info(f"Waiting for TF {obj} to become available...")
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        try:
            tf_camera_in_base = self.tf_buffer.lookup_transform(
                "fr3_link0",
                "dave_color_optical_frame",
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )

            tf_obj_in_camera = self.tf_buffer.lookup_transform(
                "dave_color_optical_frame",
                obj,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )

            T_camera_in_base = self.transform_to_matrix(tf_camera_in_base)
            T_obj_in_camera = self.transform_to_matrix(tf_obj_in_camera)

            T_obj_in_base = T_camera_in_base @ T_obj_in_camera

            # transform_stamped = self.tf_buffer.lookup_transform(
            #     obj,      # Target frame
            #     "base",      # Source frame
            #     rclpy.time.Time(),     # Get the latest available transform
            #     rclpy.duration.Duration(seconds=1.0) # Timeout
            # )
            self.obj2pose[obj] = T_obj_in_base
            
            self.get_logger().info(f"TF for {obj} successfully retrieved and stored.")

        except Exception as e:
            self.get_logger().error(f"Failed to get pose for {obj}: {e}")


    def apply_relative_transform(self, base_xyz, base_quat, rel_xyz, rel_quat):
        R_base = R.from_quat(base_quat).as_matrix()
        T_base = np.eye(4)
        T_base[:3, :3] = R_base
        T_base[:3, 3] = base_xyz

        R_rel = R.from_quat(rel_quat).as_matrix()
        T_rel = np.eye(4)
        T_rel[:3, :3] = R_rel
        T_rel[:3, 3] = rel_xyz

        T_result = T_base @ T_rel
        new_pos = T_result[:3, 3]
        new_quat = R.from_matrix(T_result[:3, :3]).as_quat()

        return new_pos, new_quat

    # ---- SKILL METHODS----
    def pick(self, obj):
        if obj not in self.obj2pose:
            self.get_logger().error(f"No pose found for {obj}, skipping pick.")
            return

        T = self.obj2pose[obj]
        self.picked_obj = obj

        offset = self.obj2offset.get(obj, np.zeros(3))

        x = T[0, 3] + offset[0]
        y = T[1, 3] + offset[1]
        z = T[2, 3] + offset[2]

        #########################################################
        #ALTERNATIVE RELATIVE GRASPING (NOT WORKING)
        
        # if obj == "spoon": 
        #     with open("spoon_grasp2.pkl","rb") as f: 
        #         rel_transform = pickle.load(f)


        # rel_transform_mat = self.transform_to_matrix(rel_transform)
        # grasp = T @ rel_transform_mat

        
        # x = grasp[0,3]
        # y = grasp[1,3] 
        # z = grasp[2,3]
        

        # rot_matrix = grasp[:3,:3]
        # rpy = R.from_matrix(rot_matrix).as_euler('xyz')

        # r = rpy[2]
        #########################################################

        rot_matrix = T[:3, :3]
        rpy = R.from_matrix(rot_matrix).as_euler('xyz')

        r = rpy[2]
        if obj =="bowl" or obj == "sponge":
            r = 0

        quaternion_xyzw = R.from_euler('XYZ', [np.pi, 0., r]).as_quat()
        self.picked_quat = quaternion_xyzw

        sys.stdin.flush()
        self.robot.pick(x, y, r, z=z, close_force=self.obj2pickparams[obj]['close_force'], close_width=self.obj2pickparams[obj]['close_width'])
        self.get_logger().info(f"Picked {obj}")

    def _place(self, x, y, r, z=None):
        print(f"placing at x:{x}, y:{y}, r{r/np.pi*180} degree")
        if z is None:
            z = self.z_min
        # if self.picked_obj != "purple_teapot":
        #     z+=0.1
        # else:
        #     z-=0.18
        #     x+=0.07
        # self.robot.place(x, y, r, z=z)

        # quaternion_xyzw = R.from_euler('XYZ', [np.pi, 0., r]).as_quat()
        # place_z_offset = 0.05
        # place_z_offset2 = 0.15

        self.robot.waypoints_goto(x, y, z+0.2, self.picked_quat)
        self.robot.waypoints_goto(x, y, z, self.picked_quat)
        self.robot.open_gripper()
        

    def place(self, obj, x_diff=0, y_diff=0, z_diff=0):
        if obj not in self.obj2pose:
            self.get_logger().error(f"No pose found for {obj}, skipping place.")
            return

        if self.picked_obj is None or self.picked_quat is None:
            self.get_logger().error("No object has been picked up yet. Skipping place.")
            return


        x,y,z = self.obj2pose[obj][:3, -1]
        
        offset = self.obj2offset.get(obj, np.zeros(3))
        
        
        x = x + x_diff + offset[0]
        y = y + y_diff + offset[1]
        z = z + z_diff + offset[2]
        r=0.0
        self._place(x, y, r, z)

        self.get_logger().info(f"Placed {obj} at its starting location.")


    def pour(self, target_obj, tilt_angle_rad=1.0, hold_time_sec=8.0):
        if target_obj not in self.obj2pose:
            self.get_pose(target_obj)

        if self.picked_obj is None or self.picked_quat is None:
            self.get_logger().error("No object has been picked up yet. Skipping pour.")
            return

        T = self.obj2pose[target_obj]
        offset = self.obj2offset.get(target_obj, np.zeros(3))

        x = T[0, 3] + offset[0] 
        y = T[1, 3] + offset[1] - 0.07
        z = T[2, 3] + offset[2] + 0.35

        # rot_matrix = T[:3, :3]
        # rpy = R.from_matrix(rot_matrix).as_euler('xyz')
        # r = rpy[2]
        # quaternion_xyzw = R.from_euler('XYZ', [np.pi, 0., r]).as_quat()

        quat = self.picked_quat

        self.robot.waypoints_goto(x, y, z, quat)
        self.get_logger().info(f"Moved above {target_obj} to pour")


        self.replay_demo(self.TILT_TEAPOT_DEMO_PATH, skip_threshold=0.03)

        # quat = self.robot.tilt_forward(current_rotation=quat, tilt_angle_rad=-1*tilt_angle_rad)
        # self.get_logger().info(f"Tilted forward to pour over {target_obj}")
        # time.sleep(hold_time_sec)

        # self.robot.tilt_forward(current_rotation=quat, tilt_angle_rad=tilt_angle_rad)
        # self.get_logger().info(f"Returned upright after pouring")

       
    def squeeze(self, squeezable, container, hold_time_sec=3.0):
        """
        Squeeze the given object inside the given container.

        This function moves the robot to hover above the container, and then
        closes the gripper to squeeze the object inside the container. The
        robot then lifts the object without diving into the container.

        Args:
            squeezable (str): The name of the object to squeeze.
            container (str): The name of the container to squeeze the object into.
            hold_time_sec (float, optional): The amount of time to hold the
                gripper closed while squeezing the object. Defaults to 3.0.
        """
        if squeezable not in self.obj2pose:
            self.get_logger().error(f"No pose found for {squeezable}, skipping squeeze.\n")
            return

        if self.picked_obj is None or self.picked_quat is None:
            self.get_logger().error("No object has been picked up yet. Skipping squeeze.\n")
            return
        
        x,y,z = self.obj2pose[container][:3, -1]

        hover_height = 0.2
        self.robot.waypoints_goto(x, y, z + hover_height, self.picked_quat)

        self.get_logger().info(f"Moved above {squeezable} to squeeze without diving.\n")

        time.sleep(0.5)
        self.robot.close_gripper()
        self.get_logger().info(f"Squeezed {squeezable} without diving.\n")

        time.sleep(hold_time_sec)

        lift_height = 0.05
        self.robot.goto_safe(x, y, z + hover_height + lift_height, self.picked_quat)
        self.get_logger().info(f"Lifted after squeezing {squeezable}.\n")


    def stack(self):
        self.replay_demo(self.STACK_DEMO_PATH, skip_threshold=0.03)
        self.open_gripper()

    def replay_demo(self, npy_fpath, skip_threshold=0.03):
        # breakpoint()
        # try:
        #     tf_base_mug_curr: TransformStamped = self.tf_buffer.lookup_transform(
        #         'base',          # target frame
        #         'white_mug',     # source frame
        #         rclpy.time.Time())  # time (latest)
        # except Exception as e:
        #     self.get_logger().error(f'Could not lookup transform: {e}')
        #     tf_base_mug_curr = None


        try:
            traj = np.load(npy_fpath, allow_pickle=True)
        except Exception as e:
            self.get_logger().error(f"Failed to load trajectory from {npy_fpath}: {e}")
            return

        self.get_logger().info(f"Loaded {len(traj)} waypoints from {npy_fpath}")
        
        prev_pose = np.zeros(7)

        for idx, stp in enumerate(traj):
            if 'franka_ee_pose_states' not in stp:
                self.get_logger().warn(f"Missing 'franka_ee_pose_states' key at step {idx}, skipping")
                continue

            pose = stp['franka_ee_pose_states']
            if not isinstance(pose, np.ndarray) or pose.shape != (7,):
                self.get_logger().warn(f"Invalid pose format at step {idx}, skipping")
                continue

            dif = pose - prev_pose
            if (abs(dif) <= skip_threshold).all():
                self.get_logger().info(f"Same pose, skipping...")
                continue
            prev_pose = pose

            #CONVERSION:
            # M_base_ee = self._pose_to_mat(pose)
            # M_mug_ee = self.M_mug_base.dot(M_base_ee)
            # # ee_pose_mug_frame = self._mat_to_pose(M_mug_ee)
            # # pose = ee_pose_mug_frame

            # M_base_mug_curr = self._tf_to_mat(tf_base_mug_curr)    # base→mug
            # M_base_ee_curr  = M_base_mug_curr.dot(M_mug_ee)  # base→EE
            # ee_pose_base_curr = self._mat_to_pose(M_base_ee_curr)
            # pose = ee_pose_base_curr
            #END CONVERSION
            
            x, y, z = pose[:3] 
            quat = pose[3:]

            self.get_logger().info(f"[{idx + 1}/{len(traj)}] → Moving to: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            # self.robot.goto_safe(x, y, z, quat)
            self.robot.waypoints_goto(x, y, z, quat)
            
        if npy_fpath == self.STACK_DEMO_PATH:
            self.open_gripper()
        elif npy_fpath == self.PICK_SPONGE_DEMO_PATH:
            self.close_gripper()
            self.replay_demo(self.RAISE_DEMO_PATH, skip_threshold=0.1)
        elif npy_fpath == self.PLACE_SPONGE_DEMO_PATH:
            self.open_gripper()
            self.replay_demo(self.RAISE_DEMO_PATH, skip_threshold=0.1)
        elif npy_fpath ==self.STACK_BOWL_DEMO_PATH_1:
            self.close_gripper()
            self.replay_demo(self.STACK_BOWL_DEMO_PATH_2, skip_threshold=0.03)
            self.open_gripper()
        elif npy_fpath ==self.STACK_BOWL_DEMO_PATH_1_TEST:
            self.close_gripper()
            self.replay_demo(self.STACK_BOWL_DEMO_PATH_2_TEST, skip_threshold=0.03)
            self.open_gripper()

        self.get_logger().info("Trajectory replay complete.")


def main():
    time.sleep(3.0)

    rclpy.init()
    node = Executor()

    while not (node.latest_rgb and node.latest_depth and node.latest_info):
        rclpy.spin_once(node, timeout_sec=0.1)

    args = sys.argv[1:]
    if len(args) < 2:
        obj1, obj2, obj3, obj4, obj5 = "cyan_cube", "sponge", "white_mug", "purple_teapot", "bowl"
    else:
        obj1, obj2 = args[0], args[1]

   
    # transform_stamped = node.tf_buffer.lookup_transform(
    #             "fr3_hand_tcp",      # Target frame
    #             "spoon",      # Source frame
    #             rclpy.time.Time(),     # Get the latest available transform
    #             rclpy.duration.Duration(seconds=1.0) # Timeout
    #         )
    # print(transform_stamped)
    # with open("spoon_grasp2.pkl", "wb") as f:
    #     pickle.dump(transform_stamped, f)
    # breakpoint()

    node.replay_demo(node.PICK_SPONGE_DEMO_PATH, skip_threshold=0.01)
    node.go_home()
    node.replay_demo(node.WIPE_DEMO_PATH, skip_threshold=0.03)
    node.go_home()
    node.replay_demo(node.PLACE_SPONGE_DEMO_PATH, skip_threshold=0.03)
    node.go_home()
    node.replay_demo(node.STACK_BOWL_DEMO_PATH_1_TEST, skip_threshold=0.03)
    node.go_home()
    node.get_pose(obj4)
    node.pick(obj4)
    node.go_home()
    node.pour(obj3)
    node.place(obj4)
    node.go_home()


    # node.replay_demo(node.STACK_BOWL_DEMO_PATH_1, skip_threshold=0.03)
    # node.open_gripper()
    # node.replay_demo(node.WIPE_DEMO_PATH, skip_threshold=0.03)

 



    # node.get_pose(obj4)
    # node.pick(obj4)
    # node.go_home()
    # node.place(obj4, x_diff=0.32, y_diff=0.05, z_diff=0.0)

    # node.replay_demo(node.STACK_BOWL_DEMO_PATH_1, skip_threshold=0.03)
    # node.go_home()
    # node.get_pose(obj5)
    # node.pick(obj5)
    # node.go_home()
    # node.place(obj5, x_diff=-0.24, y_diff=0.25, z_diff=0.2)

    # node.replay_demo(node.WIPE_DEMO_PATH, skip_threshold=0.03)
    # node.replay_demo(node.PICK_SPONGE_DEMO_PATH, skip_threshold=0.02)
    # node.go_home()


    # node.get_pose(obj3)
    # node.pick(obj3)
    # node.go_home()
    # node.place(obj3)
    # node.pour(obj3, tilt_angle_rad=1.5)

    # node.get_pose(obj4)
    # node.pick(obj4)
    # node.go_home()


    # tf_camera_in_base = node.tf_buffer.lookup_transform("fr3_hand",
    #             obj1,
    #             rclpy.time.Time(),
    #             timeout=Duration(seconds=1.0)
    #         )
    
    # print(node.transform_to_matrix(tf_camera_in_base))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
