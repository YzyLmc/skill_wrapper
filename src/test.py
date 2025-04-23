from data_structure import Skill, Predicate, PredicateState, yaml
from invent_predicate import *
from utils import GPT4

type_dict = {
    "Robot": ["robot"], 
    "Apple": ['object'], 
    "Banana": ['object'], 
    "Table": ['location'], 
    "Couch": ['location']
    }

pred = Predicate("At", ["object", "location"])
grounded_pred = pred.ground_with(["Apple", "Table"], type_dict)
lifted_pred = grounded_pred.lifted()
skill = Skill("PlaceAt", ["object", "location"])
grounded_skill = skill.ground_with(["Apple", "Table"], type_dict)
lifted_skill = grounded_skill.lifted()

lifted_pred_list = [
    Predicate("At", ["object", "location"]),
    Predicate("CloseTo", ["robot", "location"]),
    Predicate("HandOccupied", []),
    Predicate("IsHolding", ["object"]),
    Predicate("EnoughBattery", []),
    Predicate('handEmpty', [])
]

pred_state = PredicateState(lifted_pred_list)

# with open("pred_state.yaml", "w") as f:
#     yaml.dump(pred_state, f)

# with open("pred_state.yaml", "r") as f:
#     loaded_data = yaml.load(f, Loader=yaml.Loader)

groundings = possible_grounded_preds(lifted_pred_list, type_dict)
grounded_skill = skill.ground_with(["Banana", "Table"])
pred_to_update = calculate_pred_to_update(groundings, grounded_skill)

model = GPT4(engine='gpt-4o-2024-11-20')
img = ['test_imgs/empty_hand_before.png']
# grounded_skill = Skill("PickUp", ["object"], ["CyanCube"])
# grounded_pred = Predicate("IsHolding", ["object"], ["CyanCube"], 
#                           "The object is being held in the gripper now."
#                           )
# truth_value = eval_pred(model, img, grounded_skill, grounded_pred, prompt_fpath=["prompts/evaluate_pred_franka.txt"])

image_pair = [
    "test_imgs/empty_hand_before.png",
    "test_imgs/full_hand_before.png"
]
grounded_skills = [
    Skill("PickUp", ["object"], ["CyanCube"]),
    Skill("PickUp", ["object"], ["CyanCube"])
]
successes = [True, False]
lifted_pred_list = [
    Predicate("At", ["object", "location"], semantic="Object is at the location."),
    Predicate("CloseTo", ["robot", "location"], semantic="Robot is close to the location."),
    Predicate("EnoughBattery", [], semantic="Robot has enough battery."),
]
pred_type = "precond"
# new_pred = generate_pred(image_pair, grounded_skills, successes, model, lifted_pred_list, pred_type)
# breakpoint()
# dummy tasks with one task with 3 steps
dummy_tasks = {
    "dummy_0":{

        0: {"skill": None,
            "image": "test_imgs/dummy_0/0.jpg",
            "success": None
        },

        1: {
            "skill": Skill("GoTo", ["location"], ["Table"]),
            "image": "test_imgs/dummy_0/1.jpg",
            "success": True
        },

        2: {
            "skill": Skill("PickUp", ["object"], ["Apple"]),
            "image": "test_imgs/dummy_0/2.jpg",
            "success": False
        },

        3: {
            "skill": Skill("PickUp", ["object"], ["Banana"]),
            "image": "test_imgs/dummy_0/3.jpg",
            "success": True
        },
    }
}

grounded_pred_truth_value_log = {}

grounded_predicate_truth_value_log = update_empty_predicates(model, dummy_tasks, lifted_pred_list, type_dict, grounded_pred_truth_value_log)

skill2task2state = grounded_pred_log_to_skill2task2state(grounded_pred_truth_value_log, dummy_tasks)

skill2operator = {
    Skill("PickUp", ["object"]):{},
    Skill("GoTo", ["location"]):{}
}
lifted_skill = Skill("PickUp", ["object"])
mismatch_pairs = detect_mismatch(lifted_skill, skill2operator, grounded_pred_truth_value_log, dummy_tasks, type_dict, pred_type="precond")
_,_, skill2partition = partition_by_termination_n_eff(skill2task2state)
breakpoint()