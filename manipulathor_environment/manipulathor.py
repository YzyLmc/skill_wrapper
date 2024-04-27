import prior
from ai2thor.controller import Controller
from PIL import Image
# from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator
import copy
import pdb

import os
import random 

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