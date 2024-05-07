import prior
from ai2thor.controller import Controller
from PIL import Image
# from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator
import copy
import pdb
import sys
import os
import random 
import cv2
import numpy as np
import json

'''
Train Set: dataset["train"]
Train Element dataset["train"][0]
Environment Properties: type(house), house.keys(), house

CONTROLS in ProcTHOR
Image.fromarray(controller.lastevent.frame) # save image from last frame
event = controller.step(action="RotateRight")
event = controller.step(action="MoveAhead")
Image.fromarray(event.frame) #save resulting observation from action

CHANGING ENVIRONMENTS
new_house = dataset["train"][1]
controller.reset(scene=new_house)

Reference Colab: https://colab.research.google.com/drive/1Il6TqmRXOkzYMIEaOU9e4-uTDTIb5Q78#scrollTo=oxcFFLubNxti
ProcTHOR Website: https://procthor.allenai.org/

ManipulaTHOR Documentation: https://ai2thor.allenai.org/manipulathor/documentation


GENERATE AND VALIDATE NEW HOUSE
house_generator = HouseGenerator(
    split="train", seed=42, room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER
)
house, _ = house_generator.sample()
house.validate(house_generator.controller)

house.to_json("temp.json")
'''

START_ARM_BASE_HEIGHT = 0.5
ARM_BASE_MOVE_INCREMENT = 0.025
ARM_MOVE_INCREMENT = 0.025

BODY_MOVE_INCREMENT = 0.2
DELTA_TIME_INCREMENT = 0.02
ROTATION_INCREMENT =  20
FIXED_SPEED = 1

ARM_GRASPING_RADIUS = 0.1
ARM_INFLUENCE_RADIUS = 0.3

BASE_DIR = '/home/shreyas/Desktop/skill_wrapper/'

def get_term_character():
    # NOTE: Leave these imports here! They are incompatible with Windows.
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

class ManipulaTHOR:
    def __init__(self, split="train", scene_id = 0):

        self.dataset = prior.load_dataset("procthor-10k")

        if type(scene_id) is int:
            self.current_scene = self.dataset[split][scene_id]
        else:
            self.current_scene = scene_id

        self.controller = Controller(
                massThreshold = 1,
                agentMode="arm",
                scene = self.current_scene,
                snapToGrid=False,
                visibilityDistance=1.5,
                gridSize=0.25,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                renderObjectImage = True,
                width= 1280,
                height= 720,
                fieldOfView=60
            )

        
        
        self.controller.step(action="SetHandSphereRadius", radius=ARM_GRASPING_RADIUS)
        self.controller.step(action="MoveArmBase", y=START_ARM_BASE_HEIGHT,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT)

        self.object_id_to_index = {}

        for idx, obj in enumerate(self.controller.last_event.metadata["objects"]):
            self.object_id_to_index[obj['objectId']] = idx

        
        self.robot_arm_center_pos = self.controller.last_event.metadata["arm"]["handSphereCenter"]
        self.robot_pos = self.controller.last_event.metadata["agent"]["position"]
        self.robot_rot = self.controller.last_event.metadata["agent"]["rotation"]
        self.robot_arm_base_height = START_ARM_BASE_HEIGHT
        
        self.last_observation_frame = None

        self.split = split
    
    def change_scene(self, new_scene_id):

        if type(new_scene_id) is int:
            new_house = self.dataset[self.split][new_scene_id]
        elif type(new_scene_id) is str:
            new_house = new_scene_id

        self.controller.reset(scene=new_house)
        self.current_scene = new_house

        self.object_id_to_index = {}

        for idx, obj in enumerate(self.controller.last_event.metadata["objects"]):
            self.object_id_to_index[obj['objectId']] = idx

    
    def get_top_down_frame(self):
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])
        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]
        return Image.fromarray(top_down_frame)
    
    def get_last_observation_frame(self, event=None):

        if event is not None:
            observation = event.frame
        else:
            observation = controller.last_event.frame
        
        self.last_observation_frame = observation
        
        return Image.fromarray(observation)
    
    def step(self, action_name=None, **kwargs):


        self.controller.step(action_name, kwargs)

   

class AI2ThorInteractor(object):
    def __init__(
        self,
        has_object_actions=True,
        save_dir=".",
    ):
        self.has_object_actions = has_object_actions
        self.save_dir = os.path.join(BASE_DIR, save_dir)

        #create save dir if it doesn't already exist
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        self.counter = 0
        self.image_counter = 0

        #autopopulate image counter if needed
        if os.path.isdir(self.save_dir):
            self.image_counter = len(os.listdir(self.save_dir))

        self.default_interact_commands = {
            # movement
            "d": dict(action="MoveRight", moveMagnitude=BODY_MOVE_INCREMENT, returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            "a": dict(action="MoveLeft", moveMagnitude=BODY_MOVE_INCREMENT, returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            "w": dict(action="MoveAhead", moveMagnitude=BODY_MOVE_INCREMENT, returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            "s": dict(action="MoveBack", moveMagnitude=BODY_MOVE_INCREMENT, returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            # rotation of robot base
            "o": dict(action="LookUp"),
            "l": dict(action="LookDown"),
            "k": dict(action="RotateAgent", degrees=-ROTATION_INCREMENT,returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            ";": dict(action="RotateAgent", degrees=ROTATION_INCREMENT,returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            # movement of robot arm base
            "z": dict(action="MoveArmBase", y=1, speed=FIXED_SPEED, returnToStart=False, fixedDeltaTime=DELTA_TIME_INCREMENT), #move up
            "x": dict(action="MoveArmBase", y=0, speed=FIXED_SPEED, returnToStart=False, fixedDeltaTime=DELTA_TIME_INCREMENT), #move down
            # movement of robot arm
            "h": dict(action="MoveArm",position=dict(x=0, y=ARM_MOVE_INCREMENT, z=0),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move y+
            "n": dict(action="MoveArm",position=dict(x=0, y=-ARM_MOVE_INCREMENT, z=0),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move y-
            "b": dict(action="MoveArm",position=dict(x=-ARM_MOVE_INCREMENT, y=0, z=0),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move x-
            "m": dict(action="MoveArm",position=dict(x=ARM_MOVE_INCREMENT, y=0, z=0),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move x+
            ",": dict(action="MoveArm",position=dict(x=0, y=0, z=ARM_MOVE_INCREMENT),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move z+
            ".": dict(action="MoveArm",position=dict(x=0, y=0, z=-ARM_MOVE_INCREMENT),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move x+
            #grasping objects
            "e": dict(action="PickupObject"),
            "r": dict(action="ReleaseObject"),
            #opening and closing
            "t": dict(action="OpenObject", objectId = None),
            "y": dict(action="CloseObject", objectId=None),
            #toggle object
            "u": dict(action="ToggleObjectOff", objectId=None),
            "i": dict(action="ToggleObjectOn", objectId=None),
            #slice object
            "f": dict(action="SliceObject", objectId=None),
            "g": dict(action="UseUpObject", objectId=None)
            
        }

        self.last_event = None
        self.special_objects = {}

        
        

    def interact(
        self,
        manipulathor,
    ):

        if not sys.stdout.isatty():
            raise RuntimeError("controller.interact() must be run from a terminal")

        default_interact_commands = self.default_interact_commands

        self._interact_commands = default_interact_commands.copy()

        # command_message = u"Enter a Command: Move \u2190\u2191\u2192\u2193, Rotate/Look Shift + \u2190\u2191\u2192\u2193, Quit 'q' or Ctrl-C"
        # print(command_message)
        
        for a, char in self.next_interact_command():
            new_commands = {}
            command_counter = dict(counter=1)

            def add_command(cc, action, **args):
                if cc["counter"] < 15:
                    com = dict(action=action)
                    com.update(args)
                    new_commands[str(cc["counter"])] = com
                    cc["counter"] += 1

            def euclidean_distance(v1, v2):
                return np.sqrt((v2['x']-v1['x'])**2 + (v2['y']-v1['y'])**2 + (v2['z']-v1['z'])**2)

            def sphere_intersects_cuboid(sphere_center, radius, obj_center, obj_size):
                c_x, c_y, c_z = sphere_center['x'], sphere_center['y'], sphere_center['z']
                x1, x2 = obj_center['x']-obj_size['x']/2, obj_center['x'] + obj_size['x']/2
                y1, y2 = obj_center['y']-obj_size['y']/2, obj_center['y'] + obj_size['y']/2
                z1, z2 = obj_center['z']-obj_size['z']/2, obj_center['z'] + obj_size['z']/2
                
                # Check if the center of the sphere is inside the cuboid
                if (x1 <= c_x <= x2) and (y1 <= c_y <= y2) and (z1 <= c_z <= z2):
                    return True
                
                # If center of sphere is not in cuboid, find distance to closest surface on cuboid
                dx = min(abs(c_x-x1), abs(c_x - x2))
                dy = min(abs(c_y-y1), abs(c_y - y2))
                dz = min(abs(c_z-z1), abs(c_z - z2))

                # check if part of sphere (not center) inside cuboid
                if np.sqrt(dx**2 + dy**2 + dz**2) <= radius:
                    return True

                return False
    
            
            if a is None and char == "p":

                self.save_image_and_state(manipulathor.robot_arm_base_height)

                print('SAVED IMAGE AND STATE: ', self.image_counter)
                self.image_counter += 1
                continue

            if a['action'] == "PickupObject":
                if len(self.last_event.metadata['arm']['heldObjects']) > 0:
                    continue
                
            if a['action'] == "MoveArmBase":
                if char == "z":
                    manipulathor.robot_arm_base_height+=ARM_BASE_MOVE_INCREMENT
                elif char == "x":
                    manipulathor.robot_arm_base_height-=ARM_BASE_MOVE_INCREMENT
                else:
                    raise Exception("Unknown character for action MoveArmBase!")
                
                a['y'] = manipulathor.robot_arm_base_height

            elif 'openable' in self.special_objects and (a['action'] == "OpenObject" or a['action']=="CloseObject"):
                
                min_dist_obj = None; min_distance = float('inf')

                
                for obj in self.special_objects['openable']:
                    index = manipulathor.object_id_to_index[obj]

                    #obj_position = self.last_event.metadata["objects"][index]['position']
                    obj_center = self.last_event.metadata["objects"][index]['axisAlignedBoundingBox']['center']
                    obj_size = self.last_event.metadata["objects"][index]['axisAlignedBoundingBox']['size']

                    arm_position = self.last_event.metadata["arm"]["handSphereCenter"]
                    
                    if sphere_intersects_cuboid(arm_position, ARM_GRASPING_RADIUS, obj_center, obj_size) and euclidean_distance(obj_center, arm_position) < min_distance:

                        min_distance = euclidean_distance(obj_center, arm_position)
                        min_dist_obj = obj

                a['objectId'] = min_dist_obj

            elif 'togglable' in self.special_objects and (a['action'] == "ToggleObjectOff" or a['action']=="ToggleObjectOn"):
                
                min_dist_obj = None; min_distance = float('inf')

                
                for obj in self.special_objects['togglable']:
                    index = manipulathor.object_id_to_index[obj]

                    obj_position = self.last_event.metadata["objects"][index]['position']
                    arm_position = self.last_event.metadata["arm"]["handSphereCenter"]
                    
                
                    if euclidean_distance(obj_position, arm_position) <= ARM_INFLUENCE_RADIUS and euclidean_distance(obj_position, arm_position) < min_distance:

                        min_distance = euclidean_distance(obj_position, arm_position)
                        min_dist_obj = obj

                a['objectId'] = min_dist_obj
            
            elif 'sliceable' in self.special_objects and (a['action']=="SliceObject"):

                min_dist_obj = None; min_distance = float('inf')

                
                for obj in self.special_objects['sliceable']:
                    index = manipulathor.object_id_to_index[obj]

                    obj_position = self.last_event.metadata["objects"][index]['position']
                    arm_position = self.last_event.metadata["arm"]["handSphereCenter"]
                    held_object = self.last_event.metadata['arm']['heldObjects'][0] if len(self.last_event.metadata['arm']['heldObjects'])>=1 else None
                
                    if held_object is not None and 'knife' in held_object.lower() and euclidean_distance(obj_position, arm_position) <= ARM_INFLUENCE_RADIUS and euclidean_distance(obj_position, arm_position) < min_distance:

                        min_distance = euclidean_distance(obj_position, arm_position)
                        min_dist_obj = obj

                a['objectId'] = min_dist_obj

            elif 'usable' in self.special_objects and (a['action']=="UseUpObject"):
                
                min_dist_obj = None; min_distance = float('inf')

                
                for obj in self.special_objects['usable']:
                    index = manipulathor.object_id_to_index[obj]

                    #obj_position = self.last_event.metadata["objects"][index]['position']
                    obj_center = self.last_event.metadata["objects"][index]['axisAlignedBoundingBox']['center']
                    obj_size = self.last_event.metadata["objects"][index]['axisAlignedBoundingBox']['size']

                    arm_position = self.last_event.metadata["arm"]["handSphereCenter"]
                    
                    if sphere_intersects_cuboid(arm_position, ARM_GRASPING_RADIUS, obj_center, obj_size) and euclidean_distance(obj_center, arm_position) < min_distance:

                        min_distance = euclidean_distance(obj_center, arm_position)
                        min_dist_obj = obj

                a['objectId'] = min_dist_obj
            event = manipulathor.controller.step(a)
            

            # if a['action'] == "PickupObject":
            #     pdb.set_trace()

            self.last_event = copy.copy(event)
            
            visible_objects = []
            openable_objects = []
            pickupable_objects = []
            toggleable_objects = []
            sliceable_objects = []
            usable_objects = []
            

            self.counter += 1
            if self.has_object_actions:
            
                for o in event.metadata["objects"]:
                    if o["visible"]:
                        visible_objects.append(o["objectId"])
                        if o["openable"]:
                            openable_objects.append(o["objectId"])
                            if o["isOpen"]:
                                add_command(
                                    command_counter,
                                    "CloseObject",
                                    objectId=o["objectId"],
                                )
                            else:
                                add_command(
                                    command_counter,
                                    "OpenObject",
                                    objectId=o["objectId"],
                                )

                        if o["toggleable"]:
                            add_command(
                                command_counter,
                                "ToggleObjectOff",
                                objectId=o["objectId"],
                            )
                            toggleable_objects.append(o["objectId"])
                        
                        if o["sliceable"]:
                            sliceable_objects.append(o["objectId"])
                        
                        if o["canBeUsedUp"]:
                            usable_objects.append(o["objectId"])
                        if len(event.metadata["inventoryObjects"]) > 0:
                            inventoryObjectId = event.metadata["inventoryObjects"][0][
                                "objectId"
                            ]
                            if (
                                o["receptacle"]
                                and (not o["openable"] or o["isOpen"])
                                and inventoryObjectId != o["objectId"]
                            ):
                                add_command(
                                    command_counter,
                                    "PutObject",
                                    objectId=inventoryObjectId,
                                    receptacleObjectId=o["objectId"],
                                )
                                add_command(
                                    command_counter, "MoveHandAhead", moveMagnitude=0.1
                                )
                                add_command(
                                    command_counter, "MoveHandBack", moveMagnitude=0.1
                                )
                                add_command(
                                    command_counter, "MoveHandRight", moveMagnitude=0.1
                                )
                                add_command(
                                    command_counter, "MoveHandLeft", moveMagnitude=0.1
                                )
                                add_command(
                                    command_counter, "MoveHandUp", moveMagnitude=0.1
                                )
                                add_command(
                                    command_counter, "MoveHandDown", moveMagnitude=0.1
                                )
                                add_command(command_counter, "DropHandObject")

                        elif o["pickupable"]:
                            pickupable_objects.append(o["objectId"])
                            add_command(
                                command_counter, "PickupObject", objectId=o["objectId"]
                            )

            self._interact_commands = default_interact_commands.copy()
            self._interact_commands.update(new_commands)
            
            self.special_objects['visible'] = visible_objects
            self.special_objects['pickupable'] = pickupable_objects
            self.special_objects['openable'] = openable_objects
            self.special_objects['togglable'] = toggleable_objects
            self.special_objects['sliceable'] = sliceable_objects
            self.special_objects['usable'] = usable_objects

            # print("Position: {}".format(event.metadata["agent"]["position"]))
            # print(command_message)
            # print("Visible Objects:\n" + "\n".join(sorted(visible_objects)))

            skip_keys = ["action", "objectId"]
            for k in sorted(new_commands.keys()):
                v = new_commands[k]
                command_info = [k + ")", v["action"]]
                if "objectId" in v:
                    command_info.append(v["objectId"])

                for ak, av in v.items():
                    if ak in skip_keys:
                        continue
                    command_info.append("%s: %s" % (ak, av))

                print(" ".join(command_info))

    def next_interact_command(self):

        current_buffer = ""
        while True:
            commands = self._interact_commands
            curr_char = get_term_character()
            current_buffer += curr_char
            if current_buffer == "q" or current_buffer == "\x03":
                exit()
                break
            if current_buffer == "p":
                yield None, curr_char
                current_buffer = ""
            if current_buffer in commands:
                yield commands[current_buffer], curr_char
                current_buffer = ""
            else:
                match = False
                for k, v in commands.items():
                    if k.startswith(current_buffer):
                        match = True
                        break

                if not match:
                    current_buffer = ""



    def save_image_and_state(self, robot_arm_base_height):

        self.last_event
        self.save_dir
        self.image_counter

        def save_image(name, im_type, image, flip_br=False):
            

            img = image
            if flip_br:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(self.save_dir, "{}_{}.png".format(name, im_type)), img)

        def array_to_image(arr, mode=None):
            return arr

        def json_write(name, obj):
            with open(os.path.join(self.save_dir,"{}_state.json".format(name)), "w") as outfile:
                json.dump(obj, outfile, indent=4, sort_keys=True)
        
        
        # 1) Save Color Image
        array = array_to_image(self.last_event.frame)
        save_image(self.image_counter, 'color', array, flip_br=True)

        # 2) Save Instance Segmentation
        array = array_to_image(self.last_event.instance_segmentation_frame)
        save_image(self.image_counter, 'instance_seg', array, flip_br=False)

        # 3) Save Class Segmentation
        # pdb.set_trace()
        # array = array_to_image(self.last_event.semantic_segmentation_frame)
        # save_image(self.image_counter, 'semantic_seg', array, flip_br=False)

        # 4) Save Depth Data
        data = self.last_event.depth_frame
        data = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        array = array_to_image(data)
        save_image(self.image_counter, 'depth', array, flip_br=False)

        # 5) Save Metadata
        metadata = copy.copy(self.last_event.metadata)
        metadata["arm"]["arm_base_height"] = robot_arm_base_height
        metadata["special_objects"] = self.special_objects
        json_write('metadata', metadata)





def run_interactor_for_scene():
    print('Scene Options: 5, 10, 25, 35, 40, 50, 55, 65, 90, 95 or FloorPlan{X}')
    scene_num = input("Enter Scene: ")

    if not scene_num.startswith('FloorPlan'):
        scene_num = int(scene_num)
    else:
        scene_num = str(scene_num)
    

    env = ManipulaTHOR()
    env.change_scene(scene_num)
    interactor = AI2ThorInteractor(save_dir='dataset')
    
    #randomly initialize position in the scene
    positions = env.controller.step(action="GetReachablePositions").metadata["actionReturn"]
    random_position = random.choice(positions)
    env.controller.step(action="Teleport", position = random_position, rotation=dict(x=0, y=0, z=0), horizon=30, standing=True)

    #run ManipulaTHOR interactor
    interactor.interact(env)

if __name__ == "__main__":  

    run_interactor_for_scene()
   

    # for i in [5,10,25,35,40,50,55,65,90,95]:
    #     print("Scene: ", i)
    #     test.change_scene(i)

    #     positions = test.controller.step(action="GetReachablePositions").metadata["actionReturn"]

    #     for i in range(10):
    #         random_position = random.choice(positions)

    #         for y_val in [0, 90, 180, 270]:
    #             event = test.controller.step(action="Teleport", position = random_position, rotation=dict(x=0, y=y_val, z=0), horizon=30, standing=True)
    #             pdb.set_trace()


    # for i in range(0,len(test.dataset["train"]),5):

    #     pdb.set_trace()
    #     while len(unexplored_objects) > 0:

    #         for obj in unexplored_objects:

    #             all_objects.add(obj['assetId'])
            
    #             if 'children' in list(obj.keys()):
    #                 unexplored_objects += obj['children']

    #             unexplored_objects.remove(obj)
        
        
    #     print('Total Objects: ', len(all_objects))
    #     print(all_objects)

    #     pdb.set_trace()
    #     save = input('Save: [y/n]')

    #     if str(save)=='y':
    #         top_down_frame = test.get_top_down_frame()
    #         top_down_frame.show()
    #     else:
    #         continue 


    #Training Scenes: 3 iTHOR + 3 ProcTHOR
    #ProcTHOR: Scene 10, Scene 40, Scene 90
    #iTHOR: FloorPlan1, FloorPlan311, FloorPlan417


    #Testing Scenes: 3 iTHOR + 3 ProcTHOR
    #ProcTHOR: Scene 25, Scene 35, Scene 108
    #iTHOR: FloorPlan4, FloorPlan305, FloorPlan403



    #Scene 50, Scene 55, Scene 65 ^^
    #UNUSED: Scene 5,


    #potential scenes: 5, 10, 25, 35, 40, 50, 55, 65, 90, 95
    #iTHOR scenes: FloorPlan1, FloorPlan4, FloorPlan311, FloorPlan305, FloorPlan 403, FloorPlan413


    '''
    THINGS TO DO
    - Get Physics simulation working for iThor and ProcThor scenes
    - Get "Open" and "Turn On" and "Turn Off" skills working in iThor and ProcThor scenes
    - Spend 10-15 min trying to figure out lag issue
    - Test "Grab" skill for 10-15min
    '''