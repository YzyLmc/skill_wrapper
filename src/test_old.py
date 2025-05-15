# import numpy as np
# import random
# import os
# from PIL import Image

# from ai2thor.controller import Controller

# from manipula_skills import *

# def capture_obs(controller, file_prefix):
#     from PIL import Image
#     import os
#     counter = 1
#     directory = f"tasks/exps/{file_prefix.split('_')[1]}/"
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     f_list = os.listdir(directory)
#     while True:
#         screenshot_path = f"{file_prefix}_{counter}.jpg"
#         screenshot_path_suc = f"{file_prefix}_True_{counter}.jpg"
#         screenshot_path_fail = f"{file_prefix}_False_{counter}.jpg"
#         # if (not os.path.exists(f"tasks/exps/{file_prefix.split('_')[1]}/{file_prefix}_{counter}.jpg")) and (not os.path.exists(f"tasks/exps/{file_prefix.split('_')[1]}/{file_prefix}_True_{counter}.jpg")) and (not os.path.exists(f"tasks/exps/{file_prefix.split('_')[1]}/{file_prefix}_False_{counter}.jpg")):
#         if not (screenshot_path in f_list or screenshot_path_suc in f_list or screenshot_path_fail in f_list):
#             break
#         counter += 1
#     event = controller.step('Pass')
#     im = Image.fromarray(event.frame)
#     im.save(f"{directory}{screenshot_path}")
#     print(f"Screenshot saved to {screenshot_path}")
#     return f"{directory}{screenshot_path}"

# # init ai2thor controller
# controller = Controller(
#     massThreshold = 1,
#                 agentMode="arm",
#                 scene = "FloorPlan203",
#                 snapToGrid=False,
#                 visibilityDistance=1.5,
#                 gridSize=0.1,
#                 renderDepthImage=True,
#                 renderInstanceSegmentation=True,
#                 renderObjectImage = True,
#                 width= 1280,
#                 height= 720,
#                 fieldOfView=90
#             )
# event = controller.reset(scene="FloorPlan203", fieldOfView=100)
# controller.step(action="SetHandSphereRadius", radius=0.15)
# sofa_pose = {'name': 'agent', 'position': {'x': -0.1749999225139618, 'y': 0.9070531129837036, 'z': 3.083493709564209}, 'rotation': {'x': -0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': 30.00001525878906, 'isStanding': True, 'inHighFrictionArea': False}
# controller.step(
#             action = 'Teleport',
#             position = sofa_pose['position'],
#             rotation = sofa_pose['rotation'],
#             horizon = int(sofa_pose['cameraHorizon']),
#             standing = sofa_pose['isStanding']
#         )

# for obj in [obj for obj in event.metadata["objects"] if 'Chair' in obj['objectId']]:
#     event = controller.step('RemoveFromScene', objectId=obj["objectId"])

# for obj in [obj for obj in event.metadata["objects"] if 'Pencil' in obj['objectId']]:
#     event = controller.step('RemoveFromScene', objectId=obj["objectId"])

# for obj in [obj for obj in event.metadata["objects"] if 'Plate' in obj['objectId']]:
#     event = controller.step('RemoveFromScene', objectId=obj["objectId"])

# for obj in [obj for obj in event.metadata["objects"] if 'CellPhone' in obj['objectId']]:
#     event = controller.step('RemoveFromScene', objectId=obj["objectId"])

# for obj in [obj for obj in event.metadata["objects"] if 'RemoteControl' in obj['objectId']]:
#     remote = deepcopy(obj)
#     event = controller.step('RemoveFromScene', objectId=obj["objectId"])

# poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "Book" in obj['name']]
# object = "Book"
# obj = [obj for obj in event.metadata["objects"] if "Book" in obj['objectId']][0]
# poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x']-0.2, 'y': obj['position']['y'], 'z': obj['position']['z']-0.2}})
# event = controller.step('SetObjectPoses',objectPoses = poses)

# poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "TissueBox" in obj['name']]
# replace_with = 'TissueBox'# replace remotecontrol
# obj_replace_with = [obj for obj in event.metadata["objects"] if 'TissueBox' in obj['objectId']][0]
# obj = remote
# poses.append({'objectName':obj_replace_with['name'], "position":{'x': obj['position']['x'] + 0.2, 'y': obj['position']['y'], 'z': obj['position']['z']-0.4}})
# event = controller.step('SetObjectPoses',objectPoses = poses)

# poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "Bowl" in obj['name']]
# object = "Bowl"
# obj = [obj for obj in event.metadata["objects"] if "Bowl" in obj['objectId']][0]
# poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] - 0.1, 'y': obj['position']['y'], 'z': obj['position']['z']}})
# event = controller.step('SetObjectPoses',objectPoses = poses)

# poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "Vase" in obj['name']]
# object = "Vase"
# obj = [obj for obj in event.metadata["objects"] if "Vase" in obj['objectId']][0]
# poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] - 0.1, 'y': obj['position']['y'], 'z': obj['position']['z']}})
# event = controller.step('SetObjectPoses',objectPoses = poses)
# screenshot_path = capture_obs(controller, f"Before_GoTo_DiningTable_DiningTable")
# suc, event = GoTo("DiningTable", "DiningTable", controller, event)
# capture_obs(controller, f"After_GoTo_DiningTable_DiningTable_{suc}")
# os.rename(screenshot_path, screenshot_path.replace(f"Before_GoTo_DiningTable_DiningTable", f"Before_GoTo_DiningTable_DiningTable_{suc}"))
# screenshot_path = capture_obs(controller, f"Before_PickUp_Book_DiningTable")
# suc, event = PickUp("Book", "DiningTable", controller, event)
# capture_obs(controller, f"After_PickUp_Book_DiningTable_{suc}")
# os.rename(screenshot_path, screenshot_path.replace(f"Before_PickUp_Book_DiningTable", f"Before_PickUp_Book_DiningTable_{suc}"))
# screenshot_path = capture_obs(controller, f"Before_PickUp_Book_DiningTable")
# suc, event = PickUp("Book", "DiningTable", controller, event)
# capture_obs(controller, f"After_PickUp_Book_DiningTable_{suc}")
# os.rename(screenshot_path, screenshot_path.replace(f"Before_PickUp_Book_DiningTable", f"Before_PickUp_Book_DiningTable_{suc}"))
# screenshot_path = capture_obs(controller, f"Before_DropAt_Book_Sofa")
# suc, event = DropAt("Book", "Sofa", controller, event)
# capture_obs(controller, f"After_DropAt_Book_Sofa_{suc}")
# os.rename(screenshot_path, screenshot_path.replace(f"Before_DropAt_Book_Sofa", f"Before_DropAt_Book_Sofa_{suc}"))
# screenshot_path = capture_obs(controller, f"Before_PickUp_Vase_DiningTable")
# suc, event = PickUp("Vase", "DiningTable", controller, event)
# capture_obs(controller, f"After_PickUp_Vase_DiningTable_{suc}")
# os.rename(screenshot_path, screenshot_path.replace(f"Before_PickUp_Vase_DiningTable", f"Before_PickUp_Vase_DiningTable_{suc}"))
# screenshot_path = capture_obs(controller, f"Before_DropAt_Vase_Sofa")
# suc, event = DropAt("Vase", "Sofa", controller, event)
# capture_obs(controller, f"After_DropAt_Vase_Sofa_{suc}")
# os.rename(screenshot_path, screenshot_path.replace(f"Before_DropAt_Vase_Sofa", f"Before_DropAt_Vase_Sofa_{suc}"))
# screenshot_path = capture_obs(controller, f"Before_DropAt_Vase_Sofa")
# suc, event = DropAt("Vase", "Sofa", controller, event)
# capture_obs(controller, f"After_DropAt_Vase_Sofa_{suc}")
# os.rename(screenshot_path, screenshot_path.replace(f"Before_DropAt_Vase_Sofa", f"Before_DropAt_Vase_Sofa_{suc}"))
# screenshot_path = capture_obs(controller, f"Before_PickUp_Bowl_DiningTable")
# suc, event = PickUp("Bowl", "DiningTable", controller, event)
# capture_obs(controller, f"After_PickUp_Bowl_DiningTable_{suc}")
# os.rename(screenshot_path, screenshot_path.replace(f"Before_PickUp_Bowl_DiningTable", f"Before_PickUp_Bowl_DiningTable_{suc}"))

# # from utils import load_from_file
# # from symbolize import cross_assignment

# # log_data = load_from_file('tasks/log/ai2thor_5_log_6.json')

# # last_run_num = max([key for key in log_data.keys() if key.isdigit()])
# # skill2tasks, skill2operators, pred_dict, grounded_skill_dictionary = log_data[last_run_num]["skill2tasks"], log_data[last_run_num]["skill2operators"], log_data[last_run_num]["pred_dict"], log_data[last_run_num]["grounded_skill_dictionary"]

# # assigned_skill2operators = cross_assignment(skill2operators, skill2tasks, pred_dict)
# # print(assigned_skill2operators)

# # exec(
# # '''
# # import os
# # print(os.getcwd())
# # '''     
# #      )
# from main import update_tasks
# skill2tasks = {'DropAt([OBJ], [LOC])': {'DropAt_Book_Bowl_False_1': {'loc': 'Bowl', 'loc_1': '', 'loc_2': '', 'obj': 'Book', 's0': ['tasks/exps/DropAt/Before_DropAt_Book_Bowl_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Book_Bowl_False_1.jpg'], 'success': False}, 'DropAt_Book_Sofa_False_1': {'loc': 'Sofa', 'loc_1': '', 'loc_2': '', 'obj': 'Book', 's0': ['tasks/exps/DropAt/Before_DropAt_Book_Sofa_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Book_Sofa_False_1.jpg'], 'success': False}, 'DropAt_Bowl_DiningTable_False_1': {'loc': 'DiningTable', 'loc_1': '', 'loc_2': '', 'obj': 'Bowl', 's0': ['tasks/exps/DropAt/Before_DropAt_Bowl_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_DiningTable_False_1.jpg'], 'success': False}, 'DropAt_Bowl_Sofa_True_1': {'loc': 'Sofa', 'loc_1': '', 'loc_2': '', 'obj': 'Bowl', 's0': ['tasks/exps/DropAt/Before_DropAt_Bowl_Sofa_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_Sofa_True_1.jpg'], 'success': True}, 'DropAt_Bowl_Sofa_True_2': {'loc': 'Sofa', 'loc_1': '', 'loc_2': '', 'obj': 'Bowl', 's0': ['tasks/exps/DropAt/Before_DropAt_Bowl_Sofa_True_2.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_Sofa_True_2.jpg'], 'success': True}, 'DropAt_Vase_Bowl_False_1': {'loc': 'Bowl', 'loc_1': '', 'loc_2': '', 'obj': 'Vase', 's0': ['tasks/exps/DropAt/Before_DropAt_Vase_Bowl_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Vase_Bowl_False_1.jpg'], 'success': False}, 'DropAt_Vase_Bowl_False_2': {'loc': 'Bowl', 'loc_1': '', 'loc_2': '', 'obj': 'Vase', 's0': ['tasks/exps/DropAt/Before_DropAt_Vase_Bowl_False_2.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Vase_Bowl_False_2.jpg'], 'success': False}}, 'GoTo([LOC_1], [LOC_2])': {'GoTo_DiningTable_Sofa_True_1': {'loc': '', 'loc_1': 'DiningTable', 'loc_2': 'Sofa', 'obj': '', 's0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_Sofa_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_Sofa_True_1.jpg'], 'success': True}, 'GoTo_DiningTable_Sofa_True_2': {'loc': '', 'loc_1': 'DiningTable', 'loc_2': 'Sofa', 'obj': '', 's0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_Sofa_True_2.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_Sofa_True_2.jpg'], 'success': True}, 'GoTo_DiningTable_Sofa_True_3': {'loc': '', 'loc_1': 'DiningTable', 'loc_2': 'Sofa', 'obj': '', 's0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_Sofa_True_3.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_Sofa_True_3.jpg'], 'success': True}, 'GoTo_DiningTable_Sofa_True_4': {'loc': '', 'loc_1': 'DiningTable', 'loc_2': 'Sofa', 'obj': '', 's0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_Sofa_True_4.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_Sofa_True_4.jpg'], 'success': True}, 'GoTo_Sofa_DiningTable_True_1': {'loc': '', 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 's0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_1.jpg'], 'success': True}, 'GoTo_Sofa_DiningTable_True_2': {'loc': '', 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 's0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_2.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_2.jpg'], 'success': True}, 'GoTo_Sofa_DiningTable_True_3': {'loc': '', 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 's0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_3.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_3.jpg'], 'success': True}, 'GoTo_Sofa_DiningTable_True_4': {'loc': '', 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 's0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_4.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_4.jpg'], 'success': True}}, 'PickUp([OBJ], [LOC])': {'PickUp_Book_DiningTable_False_1': {'loc': 'DiningTable', 'loc_1': '', 'loc_2': '', 'obj': 'Book', 's0': ['tasks/exps/PickUp/Before_PickUp_Book_DiningTable_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Book_DiningTable_False_1.jpg'], 'success': False}, 'PickUp_Book_DiningTable_False_2': {'loc': 'DiningTable', 'loc_1': '', 'loc_2': '', 'obj': 'Book', 's0': ['tasks/exps/PickUp/Before_PickUp_Book_DiningTable_False_2.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Book_DiningTable_False_2.jpg'], 'success': False}, 'PickUp_Bowl_DiningTable_False_1': {'loc': 'DiningTable', 'loc_1': '', 'loc_2': '', 'obj': 'Bowl', 's0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_False_1.jpg'], 'success': False}, 'PickUp_Bowl_DiningTable_False_2': {'loc': 'DiningTable', 'loc_1': '', 'loc_2': '', 'obj': 'Bowl', 's0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_False_2.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_False_2.jpg'], 'success': False}, 'PickUp_Bowl_DiningTable_False_3': {'loc': 'DiningTable', 'loc_1': '', 'loc_2': '', 'obj': 'Bowl', 's0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_False_3.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_False_3.jpg'], 'success': False}, 'PickUp_Bowl_Vase_False_1': {'loc': 'Vase', 'loc_1': '', 'loc_2': '', 'obj': 'Bowl', 's0': ['tasks/exps/PickUp/Before_PickUp_Bowl_Vase_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_Vase_False_1.jpg'], 'success': False}, 'PickUp_Vase_DiningTable_True_1': {'loc': 'DiningTable', 'loc_1': '', 'loc_2': '', 'obj': 'Vase', 's0': ['tasks/exps/PickUp/Before_PickUp_Vase_DiningTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Vase_DiningTable_True_1.jpg'], 'success': True}, 'PickUp_Vase_DiningTable_True_2': {'loc': 'DiningTable', 'loc_1': '', 'loc_2': '', 'obj': 'Vase', 's0': ['tasks/exps/PickUp/Before_PickUp_Vase_DiningTable_True_2.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Vase_DiningTable_True_2.jpg'], 'success': True}}}
# skill2tasks = update_tasks(skill2tasks)
# breakpoint()
# from symbolize import cross_assignment, merge_predicates

# skill2operators = {'DropAt([OBJ], [LOC])': {'eff': {'is_within_reach([OBJ], [LOC])': 1}, 'precond': {'is_at_location([LOC])': True}}, 'GoTo([LOC_1], [LOC_2])': {'eff': {'is_at_location([LOC])': 1}, 'precond': {}}, 'PickUp([OBJ], [LOC])': {'eff': {'is_within_reach([OBJ], [LOC])': -1}, 'precond': {'is_within_reach([OBJ], [LOC])': True, 'is_clear_path([LOC])': True}}}
# skill2tasks = {'DropAt([OBJ], [LOC])': {'DropAt_TissueBox_DiningTable_False_2': {'s0': ['tasks/exps/DropAt/Before_DropAt_TissueBox_DiningTable_False_2.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_TissueBox_DiningTable_False_2.jpg'], 'success': False, 'obj': 'TissueBox', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Bowl_Sofa_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Bowl_Sofa_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_Sofa_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'DropAt_Vase_Sofa_True_2': {'s0': ['tasks/exps/DropAt/Before_DropAt_Vase_Sofa_True_2.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Vase_Sofa_True_2.jpg'], 'success': True, 'obj': 'Vase', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'DropAt_Bowl_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Bowl_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Bowl', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Book_DiningTable_False_2': {'s0': ['tasks/exps/DropAt/Before_DropAt_Book_DiningTable_False_2.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Book_DiningTable_False_2.jpg'], 'success': False, 'obj': 'Book', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Vase_Sofa_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Vase_Sofa_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Vase_Sofa_False_1.jpg'], 'success': False, 'obj': 'Vase', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'DropAt_Vase_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Vase_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Vase_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Vase', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Book_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Book_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Book_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Book', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_TissueBox_DiningTable_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_TissueBox_DiningTable_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_TissueBox_DiningTable_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Book_Sofa_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Book_Sofa_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Book_Sofa_False_1.jpg'], 'success': False, 'obj': 'Book', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'DropAt_TissueBox_Sofa_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_TissueBox_Sofa_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_TissueBox_Sofa_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}}, 'PickUp([OBJ], [LOC])': {'PickUp_TissueBox_Sofa_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'PickUp_Bowl_DiningTable_False_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Bowl', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_TissueBox_Sofa_True_2': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_True_2.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_True_2.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'PickUp_Vase_DiningTable_False_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Vase_DiningTable_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Vase_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Vase', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_Book_DiningTable_False_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Book_DiningTable_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Book_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Book', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_Vase_DiningTable_True_2': {'s0': ['tasks/exps/PickUp/Before_PickUp_Vase_DiningTable_True_2.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Vase_DiningTable_True_2.jpg'], 'success': True, 'obj': 'Vase', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}}, 'GoTo([LOC_1], [LOC_2])': {'GoTo_DiningTable_Sofa_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_Sofa_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_Sofa_True_1.jpg'], 'success': True, 'loc_1': 'DiningTable', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}, 'GoTo_DiningTable_Sofa_True_2': {'s0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_Sofa_True_2.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_Sofa_True_2.jpg'], 'success': True, 'loc_1': 'DiningTable', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}, 'GoTo_Sofa_DiningTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}, 'GoTo_Sofa_DiningTable_True_2': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_2.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_2.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}, 'GoTo_Sofa_DiningTable_True_3': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_3.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_3.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}, 'GoTo_DiningTable_Sofa_False_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_Sofa_False_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_Sofa_False_1.jpg'], 'success': False, 'loc_1': 'DiningTable', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}, 'GoTo_DiningTable_DiningTable_False_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_DiningTable_False_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_DiningTable_False_1.jpg'], 'success': False, 'loc_1': 'DiningTable', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}}}
# pred_dict = {'is_at_location([LOC])': {'semantic': 'The robot is physically present at the specified location where the object is to be dropped.', 'task': {'DropAt_Book_Bowl_False_1': [False, False], 'DropAt_Book_Sofa_False_1': [True, True], 'DropAt_Bowl_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [True, True], 'DropAt_Bowl_Sofa_True_2': [True, True], 'DropAt_Vase_Bowl_False_1': [False, False], 'DropAt_Vase_Bowl_False_2': [False, False], 'GoTo_DiningTable_Sofa_True_1': [False, True], 'GoTo_DiningTable_Sofa_True_2': [False, True], 'GoTo_DiningTable_Sofa_True_3': [False, True], 'GoTo_DiningTable_Sofa_True_4': [False, True], 'GoTo_Sofa_DiningTable_True_1': [False, True], 'GoTo_Sofa_DiningTable_True_2': [False, True], 'GoTo_Sofa_DiningTable_True_3': [False, True], 'GoTo_Sofa_DiningTable_True_4': [False, True], 'PickUp_Book_DiningTable_False_1': [True, True], 'PickUp_Book_DiningTable_False_2': [True, True], 'PickUp_Bowl_DiningTable_False_1': [False, False], 'PickUp_Bowl_DiningTable_False_2': [True, True], 'PickUp_Bowl_DiningTable_False_3': [True, True], 'PickUp_Bowl_Vase_False_1': [False, False], 'PickUp_Vase_DiningTable_True_1': [True, True], 'PickUp_Vase_DiningTable_True_2': [True, True], 'DropAt_TissueBox_DiningTable_False_2': [False, False], 'DropAt_Vase_Sofa_True_2': [True, True], 'DropAt_Book_DiningTable_False_2': [True, True], 'DropAt_Vase_Sofa_False_1': [True, True], 'DropAt_Vase_DiningTable_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [False, False], 'DropAt_TissueBox_DiningTable_True_1': [True, True], 'DropAt_TissueBox_Sofa_True_1': [True, True], 'GoTo_DiningTable_Sofa_False_1': [True, True], 'GoTo_DiningTable_DiningTable_False_1': [False, False], 'PickUp_TissueBox_Sofa_True_1': [True, True], 'PickUp_TissueBox_Sofa_True_2': [True, True], 'PickUp_Vase_DiningTable_False_1': [False, False]}}, 'is_within_reach([OBJ], [LOC])': {'semantic': "The object is within the robot's reachable distance at the specified location.", 'task': {'DropAt_Book_Bowl_False_1': [True, True], 'DropAt_Book_Sofa_False_1': [True, True], 'DropAt_Bowl_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [False, True], 'DropAt_Bowl_Sofa_True_2': [False, False], 'DropAt_Vase_Bowl_False_1': [False, False], 'DropAt_Vase_Bowl_False_2': [False, False], 'GoTo_DiningTable_Sofa_True_1': [True, True], 'GoTo_DiningTable_Sofa_True_2': [True, True], 'GoTo_DiningTable_Sofa_True_3': [True, True], 'GoTo_DiningTable_Sofa_True_4': [False, True], 'GoTo_Sofa_DiningTable_True_1': [False, True], 'GoTo_Sofa_DiningTable_True_2': [True, True], 'GoTo_Sofa_DiningTable_True_3': [False, True], 'GoTo_Sofa_DiningTable_True_4': [False, False], 'PickUp_Book_DiningTable_False_1': [True, True], 'PickUp_Book_DiningTable_False_2': [True, True], 'PickUp_Bowl_DiningTable_False_1': [False, False], 'PickUp_Bowl_DiningTable_False_2': [False, False], 'PickUp_Bowl_DiningTable_False_3': [False, False], 'PickUp_Bowl_Vase_False_1': [False, False], 'PickUp_Vase_DiningTable_True_1': [True, False], 'PickUp_Vase_DiningTable_True_2': [True, False], 'DropAt_TissueBox_DiningTable_False_2': [False, False], 'DropAt_Vase_Sofa_True_2': [True, True], 'DropAt_Book_DiningTable_False_2': [True, True], 'DropAt_Vase_Sofa_False_1': [True, False], 'DropAt_Vase_DiningTable_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [False, False], 'DropAt_TissueBox_DiningTable_True_1': [False, True], 'DropAt_TissueBox_Sofa_True_1': [False, True], 'GoTo_DiningTable_Sofa_False_1': [True, True], 'GoTo_DiningTable_DiningTable_False_1': [False, False], 'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_TissueBox_Sofa_True_2': [True, False], 'PickUp_Vase_DiningTable_False_1': [False, False]}}, 'is_clear_path([LOC])': {'task': {'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Bowl_DiningTable_False_1': [False, False], 'PickUp_TissueBox_Sofa_True_2': [True, False], 'PickUp_Vase_DiningTable_False_1': [False, True], 'PickUp_Book_DiningTable_False_1': [False, False], 'PickUp_Vase_DiningTable_True_2': [True, True], 'DropAt_TissueBox_DiningTable_False_2': [False, False], 'DropAt_Bowl_Sofa_True_1': [True, True], 'DropAt_Vase_Sofa_True_2': [True, True], 'DropAt_Bowl_DiningTable_False_1': [False, False], 'DropAt_Book_DiningTable_False_2': [True, True], 'DropAt_Vase_Sofa_False_1': [True, True], 'DropAt_Vase_DiningTable_False_1': [True, True], 'DropAt_Book_DiningTable_False_1': [True, False], 'DropAt_TissueBox_DiningTable_True_1': [True, True], 'DropAt_Book_Sofa_False_1': [True, True], 'DropAt_TissueBox_Sofa_True_1': [True, True], 'GoTo_DiningTable_Sofa_True_1': [False, True], 'GoTo_DiningTable_Sofa_True_2': [False, False], 'GoTo_Sofa_DiningTable_True_1': [False, False], 'GoTo_Sofa_DiningTable_True_2': [True, True], 'GoTo_Sofa_DiningTable_True_3': [True, True], 'GoTo_DiningTable_Sofa_False_1': [True, False], 'GoTo_DiningTable_DiningTable_False_1': [False, True]}, 'semantic': "There are no obstacles blocking the robot's path to the location of the object."}}
# assigned_skill2operators = cross_assignment(skill2operators, skill2tasks, pred_dict)
# print(assigned_skill2operators)

# skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {}, 'eff': {}}, 'DropAt([OBJ], [LOC])': {'precond': {'isHolding([OBJ])': True}, 'eff': {}}, 'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}}
# skill2tasks = {'DropAt([OBJ], [LOC])': {'DropAt_Vase_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Vase_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Vase_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Vase', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Bowl_CoffeeTable_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Bowl_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_CoffeeTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'CoffeeTable', 'loc_1': '', 'loc_2': ''}}, 'GoTo([LOC_1], [LOC_2])': {'GoTo_DiningTable_CoffeeTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_CoffeeTable_True_1.jpg'], 'success': True, 'loc_1': 'DiningTable', 'loc_2': 'CoffeeTable', 'obj': '', 'loc': ''}, 'GoTo_Sofa_DiningTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}, 'GoTo_CoffeeTable_Sofa_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_CoffeeTable_Sofa_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_CoffeeTable_Sofa_True_1.jpg'], 'success': True, 'loc_1': 'CoffeeTable', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}}, 'PickUp([OBJ], [LOC])': {'PickUp_TissueBox_Sofa_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'PickUp_Vase_CoffeeTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Vase_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Vase_CoffeeTable_True_1.jpg'], 'success': True, 'obj': 'Vase', 'loc': 'CoffeeTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_Bowl_DiningTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}}}
# pred_dict = {'isHolding([OBJ])': {'task': {'GoTo_DiningTable_CoffeeTable_True_1': [False, False], 'GoTo_Sofa_DiningTable_True_1': [False, False], 'GoTo_CoffeeTable_Sofa_True_1': [False, False], 'PickUp_TissueBox_Sofa_True_1': [False, False], 'PickUp_Vase_CoffeeTable_True_1': [True, False], 'PickUp_Bowl_DiningTable_True_1': [False, True], 'DropAt_Bowl_CoffeeTable_True_1': [True, False], 'DropAt_Vase_DiningTable_False_1': [False, False]}, 'semantic': 'The robot is currently holding the object.'}}
# merged_skill2operators, equal_preds = merge_predicates(model, skill2operators, pred_dict)
# assigned_skill2operators = cross_assignment(merged_skill2operators, skill2tasks, pred_dict, equal_preds=equal_preds)
# print(assigned_skill2operators)


# def update_replay_buffer(replay_buffer, chosen_skill_sequence, pred_dict, skill2tasks, old_skill2tasks):
#     'update replay buffer after one iteration'
#     # calculate new tasks
#     def convert_to_skill(command):
#         if command.startswith("GoTo"):
#             args = command[5:-1].replace(' ','').split(",")  # Extract arguments
#             return f'GoTo_{args[0]}_{args[1]}'
#         elif command.startswith("PickUp"):
#             args = command[7:-1].replace(' ','').split(",")  # Extract arguments
#             return f'PickUp_{args[0]}_{args[1]}'
#         elif command.startswith("DropAt"):
#             args = command[7:-1].replace(' ','').split(",")  # Extract arguments
#             return f'DropAt_{args[0]}_{args[1]}'

#     # breakpoint()
#     if not 'num2id' in replay_buffer:
#         replay_buffer['num2id'] = {}
#     new_tasks = {}
#     for s, tasks in skill2tasks.items():
#         for id, task in tasks.items():
#             if not id in old_skill2tasks[s]:
#                 new_tasks[id] = task

#     replay_buffer['skill'].extend(chosen_skill_sequence)

#     for command in chosen_skill_sequence:
#         skill_prefix = convert_to_skill(command)
#         for id, t in new_tasks.items():
#             if skill_prefix in id:
#                 replay_buffer['image_before'].append(t['s0'][0])
#                 replay_buffer['image_after'].append(t['s1'][0])
#                 # len_before = len(replay_buffer['skill']) - len(chosen_skill_sequence)
#                 len_before = len(replay_buffer['num2id'])
#                 replay_buffer['num2id'][len(replay_buffer['num2id'])] = id
#                 replay_buffer['num2id'][len(replay_buffer['num2id'])] = id

#     breakpoint()
#     predicate_eval = []
#     for i in range(len(replay_buffer['skill'])):
#         truth_values = []
#         for p in pred_dict:
#             idx = replay_buffer['num2id'][i]
#             truth_values.append(pred_dict[p]['task'][idx][0])
#             truth_values.append(pred_dict[p]['task'][idx][1])
#         predicate_eval.append(truth_values)
#     replay_buffer['predicate_eval'] = predicate_eval

#     print(replay_buffer)
#     return replay_buffer


# replay_buffer = {'image_before': [], 'image_after': [], 'skill': [], 'predicate_eval': []}
# chosen_skill_sequence = ['GoTo(Sofa,CoffeeTable)', 'PickUp(Vase,CoffeeTable)', 'GoTo(CoffeeTable,DiningTable)', 'DropAt(Vase,DiningTable)', 'PickUp(Bowl,DiningTable)', 'DropAt(Bowl,Sofa)', 'PickUp(TissueBox,Sofa)', 'GoTo(Sofa,CoffeeTable)']
# pred_dict = {'is_within_reach([OBJ], [LOC])': {'task': {'GoTo_Sofa_CoffeeTable_False_2': [False, False], 'GoTo_Sofa_CoffeeTable_True_1': [False, False], 'PickUp_Bowl_DiningTable_True_1': [True, False], 'PickUp_TissueBox_Sofa_False_2': [False, False]}, 'semantic': "The object is within the robot's reachable distance from the specified location."}, 'path_clear([LOC_1], [LOC_2])': {'task': {'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_TissueBox_Sofa_False_2': [False, False], 'GoTo_Sofa_CoffeeTable_True_1': [True, False], 'GoTo_Sofa_CoffeeTable_False_2': [False, True]}, 'semantic': 'There are no obstacles blocking the direct path from the initial position to the goal position.'}}
# skill2tasks = {'DropAt([OBJ], [LOC])': {}, 'GoTo([LOC_1], [LOC_2])': {'GoTo_Sofa_CoffeeTable_False_2': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_CoffeeTable_False_2.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_CoffeeTable_False_2.jpg'], 'success': False, 'loc_1': 'Sofa', 'loc_2': 'CoffeeTable', 'obj': '', 'loc': ''}, 'GoTo_Sofa_CoffeeTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_CoffeeTable_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'CoffeeTable', 'obj': '', 'loc': ''}}, 'PickUp([OBJ], [LOC])': {'PickUp_Bowl_DiningTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_TissueBox_Sofa_False_2': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_False_2.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_False_2.jpg'], 'success': False, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}}}
# old_skill2tasks = {'DropAt([OBJ], [LOC])': {}, 'GoTo([LOC_1], [LOC_2])': {}, 'PickUp([OBJ], [LOC])': {'PickUp_Bowl_DiningTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_TissueBox_Sofa_False_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_False_1.jpg'], 'success': False, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}}}


# replay_buffer = update_replay_buffer(replay_buffer, chosen_skill_sequence, pred_dict, skill2tasks, old_skill2tasks)
# breakpoint()
from collections import defaultdict
from symbolize import score, partition_by_effect
from copy import deepcopy
from utils import load_from_file
def convert_to_task2pred(pred_dict):
    task2pred = {}
    for pred, content in pred_dict.items():
        tasks = content['task']
        for t, values in tasks.items():
            if not t in task2pred:
                task2pred[t] = {}
            task2pred[t][pred] = values
    return task2pred

log_data = load_from_file('tasks/log/ai2thor_5_log_50.json')
last_run_num = '5'

from main import complete_grounded_skill_dict
skill2tasks, skill2operators, pred_dict, grounded_skill_dictionary, replay_buffer = log_data[last_run_num]["skill2tasks"], log_data[last_run_num]["skill2operators"], log_data[last_run_num]["pred_dict"], log_data[last_run_num]["grounded_skill_dictionary"], log_data[last_run_num]["replay_buffer"]
new_grounded_skill_dictionary= {
            "DropAt(obj, loc)": {
                "arguments": {
                    "loc": "the receptacle onto which object is dropped",
                    "obj": "the object to be dropped"
                },
                "effects_negative": [],
                "effects_positive": [],
                "preconditions": []
            },
            "GoTo(init, goal)": {
                "arguments": {
                    "init": "the location for the robot to start from",
                    "to": "the location for the robot to go to"
                },
                "effects_negative": [],
                "effects_positive": [],
                "preconditions": []
            },
            "PickUp(obj, loc)": {
                "arguments": {
                    "loc": "the receptacle that the object is picked up from",
                    "obj": "the object to be picked up"
                },
                "effects_negative": [],
                "effects_positive": [],
                "preconditions": []
            }
        }

grounded_skill_dictionary = {
                'PickUp(obj, loc)':{'arguments': {'obj': "the object to be picked up", "loc": "the receptacle that the object is picked up from"}, 'preconditions': [],  'effects_positive':[], 'effects_negative': []},
                'DropAt(obj, loc)': {'arguments': {'obj': "the object to be dropped", 'loc': "the receptacle onto which object is dropped"}, 'preconditions': [], 'effects_positive':[], 'effects_negative': []},
                'GoTo(init, goal)': {'arguments': {'init': "the location for the robot to start from", 'to': "the location for the robot to go to"}, 'preconditions': [], 'effects_positive':[], 'effects_negative':[]}
            }

output = complete_grounded_skill_dict(grounded_skill_dictionary, new_grounded_skill_dictionary)
breakpoint()
# from symbolize import partition_by_effect
# partitioned_output = partition_by_effect(pred_dict)
# grounded_skill_dictionary = defaultdict(dict)
# for idx, operator in partitioned_output.items():
#     base_action_name = operator['name'][0].split('_')[0]
#     action_counter = 1
#     # breakpoint()
#     # hardcode skill name, will remove after workshop
#     if "PickUp" in base_action_name or "DropAt" in base_action_name:
#         action_name = f"{base_action_name}_{action_counter}(obj, loc)"
#     else:
#         action_name = f"{base_action_name}_{action_counter}(init, goal)"
#     while action_name in grounded_skill_dictionary:
#         action_counter += 1
#         if "PickUp" in base_action_name or "DropAt" in base_action_name:
#             action_name = f"{base_action_name}_{action_counter}(obj, loc)"
#         else:
#             action_name = f"{base_action_name}_{action_counter}(init, goal)"

#     grounded_skill_dictionary[action_name]['precondition'] = [p.replace('([OBJ]', '(obj').replace('[OBJ])', 'obj)').replace('([LOC_1]', '(init').replace('[LOC_2])', 'goal)').replace('([LOC]', '(loc').replace('[LOC])', 'loc)') for p, value in operator['precondition'].items() if value == 1]
#     grounded_skill_dictionary[action_name]['effect'] = {p.replace('([OBJ]', '(obj').replace('[OBJ])', 'obj)').replace('([LOC_1]', '(init').replace('[LOC_2])', 'goal)').replace('([LOC]', '(loc').replace('[LOC])', 'loc)'):value for p, value in operator['effect'].items()}

# from main import complete_grounded_skill_dict
# old_ground_skill_dictionary = {
#                 'PickUp(obj, loc)':{'arguments': {'obj': "the object to be picked up", "loc": "the receptacle that the object is picked up from"}, 'preconditions': [],  'effects_positive':[], 'effects_negative': []},
#                 'DropAt(obj, loc)': {'arguments': {'obj': "the object to be dropped", 'loc': "the receptacle onto which object is dropped"}, 'preconditions': [], 'effects_positive':[], 'effects_negative': []},
#                 'GoTo(init, goal)': {'arguments': {'init': "the location for the robot to start from", 'to': "the location for the robot to go to"}, 'preconditions': [], 'effects_positive':[], 'effects_negative':[]}
#             }
            
# final_dict = complete_grounded_skill_dict(old_ground_skill_dictionary, grounded_skill_dictionary)
# breakpoint()
# task2pred = convert_to_task2pred(pred_dict)
# breakpoint()


filtered_pred = []

for pred in pred_dict:
    for skill in skill2operators:
        tscore_eff, fscore_eff = score(pred, skill, skill2tasks, pred_dict, [], 'eff')
        tscore_precond, fscore_precond = score(pred, skill, skill2tasks, pred_dict, [], 'precond')
        if (tscore_precond > 0.5 and  fscore_precond > 0.5) or (abs(tscore_eff) > 0.2 and fscore_eff > 0.2):
            filtered_pred.append(pred)
filtered_pred = list(set(filtered_pred))
# breakpoint()
pred_dict_filtered = {}
for pred, value in pred_dict.items():
    if pred in filtered_pred:
        pred_dict_filtered[pred] = value

def group_nested_dict(nested_dict):
        'calculate precondition and effect based on task2pred'
        'Effect is calculated as change of truth values'
        'Preconddtion is the intersection of '
        result = defaultdict(lambda: {'name': [], 'effect': {}, 'precond': {}})
        for outer_key, inner_dict in nested_dict.items():
            value_dict = {k: v[1] - v[0] for k, v in inner_dict.items() if v[1] - v[0] != 0}
            group_id = None
            for gid, group in result.items():
                if group['effect'] == value_dict:
                    group_id = gid
                    break
            if group_id is None:
                group_id = len(result)
                result[group_id]['effect'] = value_dict
            result[group_id]['name'].append(outer_key)
            for k, v in inner_dict.items():
                if k not in result[group_id]['precond']:
                    result[group_id]['precond'][k] = v[0]
                else:
                    if result[group_id]['precond'][k] != v[0]:
                        result[group_id]['precond'].pop(k)
        return dict(result)

# # Example usage
# nested_dict = {
#     'apple': {'a': [1, 0], 'b': [1, 1], 'c': [0, 1]},
#     'banana': {'a': [1, 0], 'b': [0, 0], 'c': [0, 1]},
#     'orange': {'a': [1, 0], 'b': [0, 1], 'c': [0, 1]}
# }

# grouped_output_with_init = group_nested_dict_with_init(nested_dict)
# print(grouped_output_with_init)


# # Example usage
# nested_dict = {
#     'apple': {'a': [1, 0], 'b': [1, 1], 'c': [0, 1]},
#     'banana': {'a': [1, 0], 'b': [0, 0], 'c': [0, 1]},
#     'orange': {'a': [1, 0], 'b': [0, 1], 'c': [0, 1]}
# }
task2pred = convert_to_task2pred(pred_dict_filtered)
suc_task2pred = {}
for task, value in task2pred.items():
    if not "False" in task:
        suc_task2pred[task] = value
grouped_output = group_nested_dict(suc_task2pred)
print(grouped_output)
partitioned_results = partition_by_effect(pred_dict)
a = 0
for i, g in grouped_output.items(): 
    a += len(g['name'])
# breakpoint()

from demonstration import generate_pddl_domain

# operators = ''
# action_counter = defaultdict(int)

# for i, skill in grouped_output.items():
#     action_name, operator= dict_to_pddl(skill, action_counter)
#     operators += operator + '\n'
output = generate_pddl_domain('exp_50', grouped_output)
breakpoint()

import re
from collections import defaultdict

# Initialize a counter for each action type
action_counter = defaultdict(int)

def extract_parameters_from_predicates(predicates):
    # Extract parameters dynamically from predicates
    parameters = set()
    for predicate in predicates:
        # Find all argument placeholders in the predicate and convert them to PDDL variables
        obj_match = re.findall(r'\[OBJ\]', predicate)
        loc_match = re.findall(r'\[LOC\]', predicate)
        loc1_match = re.findall(r'\[LOC_1\]', predicate)
        loc2_match = re.findall(r'\[LOC_2\]', predicate)

        if obj_match:
            parameters.add('?i - item')
        if loc_match:
            parameters.add('?l - location')
        if loc1_match:
            parameters.add('?l1 - location')
        if loc2_match:
            parameters.add('?l2 - location')

    return ' '.join(parameters)

def extract_arguments(action_dict):
    # Gather all predicates from preconditions and effects
    all_predicates = list(action_dict['precond'].keys()) + list(action_dict['effect'].keys())
    # Extract parameters from predicates
    return extract_parameters_from_predicates(all_predicates)

def dict_to_pddl(action_dict):
    # Extract base action name
    base_action_name = action_dict['name'][0].split('_')[0]

    # Update the counter for this action type
    action_counter[base_action_name] += 1

    # Formulate the indexed action name
    indexed_action_name = f"{base_action_name}_{action_counter[base_action_name]}"

    # Extract parameters dynamically based on the arguments of the predicates
    parameters = extract_arguments(action_dict)

    # Initialize preconditions and effects lists
    preconditions = []
    effects = []

    # Process preconditions
    for precond, value in action_dict['precond'].items():
        pddl_precond = precond.replace('[OBJ]', '?i').replace('[LOC]', '?l').replace('[LOC_1]', '?l1').replace('[LOC_2]', '?l2')
        if value:
            preconditions.append(f"({pddl_precond.lower()})")
        else:
            preconditions.append(f"(not ({pddl_precond.lower()}))")

    # Process effects
    for effect, value in action_dict['effect'].items():
        pddl_effect = effect.replace('[OBJ]', '?i').replace('[LOC]', '?l')
        if value == 1:
            effects.append(f"({pddl_effect.lower()})")
        elif value == -1:
            effects.append(f"(not ({pddl_effect.lower()}))")

    # Construct the PDDL action statement
    pddl_statement = f"(:action {indexed_action_name}\n"
    pddl_statement += f"    :parameters ({parameters})\n"
    pddl_statement += "    :precondition (and " + " ".join(preconditions) + ")\n"
    pddl_statement += "    :effect (and " + " ".join(effects) + ")\n"
    pddl_statement += ")"
    
    return pddl_statement

# Convert to PDDL
pddl_output1 = dict_to_pddl(grouped_output)
print(pddl_output1)




# from itertools import product

# # Define objects and locations
# objects = ['a', 'b', 'c']
# containers = ['A', 'B', 'C']

# # Generate all possible states (robot location, object locations, robot holding state)
# # States are tuples: (robot_location, object_a_location, object_b_location, object_c_location, holding)
# states = list(product(containers, containers + ['holding'], containers + ['holding'], containers + ['holding'], ['hand_empty', 'holding']))

# from itertools import product

# # Define objects and locations
# objects = ['a', 'b', 'c']
# containers = ['A', 'B', 'C']

# # Generate all possible states (robot location, object locations, robot holding state)
# # States are tuples: (robot_location, object_a_location, object_b_location, object_c_location, holding)
# states = []
# for robot_loc, obj_a_loc, obj_b_loc, obj_c_loc in product(containers, containers + ['holding'], containers + ['holding'], containers + ['holding']):
#     # Ensure that at most one object is 'holding'
#     held_objects = [obj_a_loc == 'holding', obj_b_loc == 'holding', obj_c_loc == 'holding']
#     if sum(held_objects) <= 1:  # Only one object can be held at a time
#         if any(held_objects):  # If an object is being held, robot can't be 'hand_empty'
#             holding = 'holding'
#         else:
#             holding = 'hand_empty'
#         states.append((robot_loc, obj_a_loc, obj_b_loc, obj_c_loc, holding))

# # Function to filter states based on predicates
# def get_states_satisfying_predicates(at=None, at_obj=None, holding=None, hand_empty=None):
#     satisfying_states = []
    
#     for state in states:
#         robot_loc, obj_a_loc, obj_b_loc, obj_c_loc, robot_holding = state

#         # Check At(loc)
#         if at and robot_loc != at:
#             continue

#         # Check AtObj(obj, loc)
#         if at_obj:
#             obj, loc = at_obj
#             if obj == 'a' and obj_a_loc != loc:
#                 continue
#             if obj == 'b' and obj_b_loc != loc:
#                 continue
#             if obj == 'c' and obj_c_loc != loc:
#                 continue

#         # Check Holding(obj)
#         if holding:
#             if robot_holding != 'holding' or (holding == 'a' and obj_a_loc != 'holding') or (holding == 'b' and obj_b_loc != 'holding') or (holding == 'c' and obj_c_loc != 'holding'):
#                 continue

#         # Check HandEmpty()
#         if hand_empty is not None and (hand_empty and robot_holding != 'hand_empty'):
#             continue

#         # If all checks pass, add the state
#         satisfying_states.append(state)

#     return satisfying_states

# # Example usage:
# # Find all states where the robot is at A, and is holding object 'a'
# example_states = get_states_satisfying_predicates(at='A', holding='a')
