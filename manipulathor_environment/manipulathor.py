import prior
from ai2thor.controller import Controller
from PIL import Image
# from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator
import copy
import pdb

import os
import random 
import cv2

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
        self.current_scene = self.dataset[split][scene_id]

        self.controller = Controller(
                massThreshold = 1,
                agentMode="arm",
                scene = self.current_scene,
                snapToGrid=False,
                visibilityDistance=1.5,
                gridSize=0.25,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                width= 1280,
                height= 720,
                fieldOfView=60
            )

        
        
        self.controller.step(action="SetHandSphereRadius", radius=0.1)
        self.controller.step(action="MoveArmBase", y=START_ARM_BASE_HEIGHT,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT)


        self.robot_arm_center_pos = self.controller.last_event.metadata["arm"]["handSphereCenter"]
        self.robot_pos = self.controller.last_event.metadata["agent"]["position"]
        self.robot_rot = self.controller.last_event.metadata["agent"]["rotation"]
        self.robot_arm_base_height = START_ARM_BASE_HEIGHT
        
        self.last_observation_frame = None

        self.split = split
    
    def change_scene(self, new_scene_id):
        new_house = self.dataset[self.split][new_scene_id]
        self.controller.reset(scene=new_house)
        self.current_scene = new_house

    
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


        self.controller.step(action=action_name, kwargs)

   

class AI2ThorInteractor(object):
    def __init__(
        self,
        has_object_actions=True,
        save_dir=".",
    ):
        self.has_object_actions = has_object_actions
        self.save_dir = save_dir
        self.counter = 0
        self.image_counter = 0

        self.default_interact_commands = {
            # movement
            "d": dict(action="MoveRight", moveMagnitude=BODY_MOVE_INCREMENT, returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            "a": dict(action="MoveLeft", moveMagnitude=BODY_MOVE_INCREMENT, returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            "w": dict(action="MoveAhead", moveMagnitude=BODY_MOVE_INCREMENT, returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            "s": dict(action="MoveBack", moveMagnitude=BODY_MOVE_INCREMENT, returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            # rotation of robot base
            "o": dict(action="LookUp"),
            "l": dict(action="LookDown"),
            "k": dict(action="RotateAgent", degrees=ROTATION_INCREMENT,returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            ";": dict(action="RotateAgent", degrees=-ROTATION_INCREMENT,returnToStart=False,speed=1,fixedDeltaTime=DELTA_TIME_INCREMENT),
            # movement of robot arm base
            "z": dict(action="MoveArmBase", y=1, speed=FIXED_SPEED, returnToStart=False, fixedDeltaTime=DELTA_TIME_INCREMENT), #move up
            "x": dict(action="MoveArmBase", y=0, speed=FIXED_SPEED, returnToStart=False, fixedDeltaTime=DELTA_TIME_INCREMENT), #move down
            # movement of robot arm
            "h": dict(action="MoveArm",position=dict(x=0, y=ARM_MOVE_INCREMENT, z=0),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move y+
            "n": dict(action="MoveArm",position=dict(x=0, y=-ARM_MOVE_INCREMENT, z=0),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move y-
            "b": dict(action="MoveArm",position=dict(x=ARM_MOVE_INCREMENT, y=0, z=0),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move x-
            "m": dict(action="MoveArm",position=dict(x=-ARM_MOVE_INCREMENT, y=0, z=0),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move x+
            ",": dict(action="MoveArm",position=dict(x=0, y=0, z=ARM_MOVE_INCREMENT),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move z+
            ".": dict(action="MoveArm",position=dict(x=0, y=0, z=-ARM_MOVE_INCREMENT),coordinateSpace="wrist",restrictMovement=True,speed=FIXED_SPEED,returnToStart=False,fixedDeltaTime=DELTA_TIME_INCREMENT), #move x+
            #grasping objects
            "g": dict(action="PickupObject"),
            "r": dict(action="ReleaseObject")
        }

        self.last_event = None

        

    def interact(
        self,
        controller,
        semantic_segmentation_frame=False,
        instance_segmentation_frame=False,
        depth_frame=False,
        color_frame=False,
        metadata=False,
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
            
            if a is None and char == "p":

                self.save_image_and_state()

                print('SAVED IMAGE AND STATE: ', self.image_counter)
                self.image_counter += 1
                continue

            if a['action'] == "MoveArmBase":
                if char == "z":
                    controller.robot_arm_base_height+=ARM_BASE_MOVE_INCREMENT
                elif char == "x":
                    controller.robot_arm_base_height-=ARM_BASE_MOVE_INCREMENT
                else:
                    raise Exception("Unknown character for action MoveArmBase!")

                a['action']['y'] = controller.robot_arm_base_height

            event = controller.step(a)
            self.last_event = copy(event)
            visible_objects = []

            self.counter += 1
            if self.has_object_actions:
                for o in event.metadata["objects"]:
                    if o["visible"]:
                        visible_objects.append(o["objectId"])
                        if o["openable"]:
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
                            add_command(
                                command_counter, "PickupObject", objectId=o["objectId"]
                            )

            self._interact_commands = default_interact_commands.copy()
            self._interact_commands.update(new_commands)

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



    def save_image_and_state(self):

        self.last_event
        self.save_dir
        self.image_counter

        def save_image(name, im_type, image, flip_br=False):
            import cv2

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
        array = array_to_image(self.last_event.semantic_segmentation_frame)
        save_image(self.image_counter, 'semantic_seg', array, flip_br=False)

        # 4) Save Depth Data
        data = self.last_event.depth_frame
        data = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        array = array_to_image(data)
        save_image(self.image_counter, 'depth', array, flip_br=False)

        # 5) Save Metadata
        json_write('metadata', self.last_event.metadata)

if __name__ == "__main__":   

    test = ManipulaTHOR()
    for i in [5,10,25,35,40,50,55,65,90,95]:
        print("Scene: ", i)
        test.change_scene(i)

        positions = test.controller.step(action="GetReachablePositions").metadata["actionReturn"]

        for i in range(10):
            random_position = random.choice(positions)

            for y_val in [0, 90, 180, 270]:
                event = test.controller.step(action="Teleport", position = random_position, rotation=dict(x=0, y=y_val, z=0), horizon=30, standing=True)
                pdb.set_trace()


    # for i in range(0,len(test.dataset["train"]),5):

    #     print("Scene: ", i)
    #     test.change_scene(i)
    #     print(test.current_scene['rooms'])

    #     all_objects =  set([])
    #     unexplored_objects = copy.copy(test.current_scene['objects'])

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

    
    #potential scenes: 5, 10, 25, 35, 40, 50, 55, 65, 90, 95