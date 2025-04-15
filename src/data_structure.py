import itertools
from copy import deepcopy

class Skill:
    def __init__(self, name, types, params=[]):
        self.name = name
        self.types = tuple(types)
        self.params = tuple(params)
    
    def __str__(self):
        param_str = ", ".join(map(str, self.params))
        type_str = ", ".join(map(str, self.types))
        return f"{self.name}({param_str})" if self.params else f"{self.name}({type_str})"
    
    def __hash__(self):
        return hash((self.name, self.types, self.params))
    
    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return (self.name, self.types, self.params) == (other.name, other.types, other.params)

    def is_grounded(self):
        return bool(self.params)
    
    @staticmethod
    def ground_with_params(lifted_skill, params: list[str], type_dict=None):
        """
        Grounded a skill or a predicate with parameters and their types.
        lifted_skill :: Skill object
        params :: list:: list of parameters, e.g., ["Apple", "Table"]
        type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
        """
        # tuple is applicable to the lifted representation
        if type_dict:
            for i, p in enumerate(params):
                assert p in type_dict
                assert lifted_skill.types[i] in type_dict[p]

        # grounded skill/predicate
        grounded_pred = deepcopy(lifted_skill)
        grounded_pred.params = tuple(params)

        return grounded_pred

class Predicate:
    def __init__(self, name, types, params=None, semantic=None):
        self.name = name
        self.types = tuple(types)
        self.params = tuple(params) if params else ()
        self.semantic = semantic
        # self.truth_value = None

    def __str__(self):
        param_str = ", ".join(map(str, self.params))
        type_str = ", ".join(map(str, self.types))
        return f"{self.name}({param_str})" if self.params else f"{self.name}({type_str})"

    def __hash__(self):
        # Use name, types, and params as hash — exclude semantic & truth value
        return hash((self.name, self.types, self.params))

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return (self.name, self.types, self.params) == (other.name, other.types, other.params)

    def is_grounded(self):
        return bool(self.params)
    
    @staticmethod
    def ground_with_params(lifted_pred, params: list[str] | tuple, type_dict=None):
        """
        Grounded a skill or a predicate with parameters and their types.
        lifted_pred :: Predicate object
        params :: list:: list of parameters, e.g., ["Apple", "Table"]
        type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
        """
        assert not grounded_pred.is_grounded(), "Cannot ground an already grounded predicate"
        # tuple is applicable to the lifted representation
        if type_dict:
            for i, p in enumerate(params):
                assert p in type_dict
                assert lifted_pred.types[i] in type_dict[p]

        # grounded predicate
        grounded_pred = deepcopy(lifted_pred)
        grounded_pred.params = tuple(params)

        return grounded_pred

    @staticmethod
    def lift_grounded_pred(grounded_pred, type_dict=None):
        """
        Lift a grounded predicate, e.g., {'name': 'At', 'types': ["object", "location"], 'params': ['Apple', 'Table']} to a {'name':"At", 'types':["object", "location"], 'params':[]}
        type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
        """
        assert grounded_pred.is_grounded(), "Cannot lift an ungrounded predicate"
        if type_dict:
            assert all([type in type_dict[param] for type, param in zip(grounded_pred.types, grounded_pred.params)])
        
                # grounded predicate
        grounded_pred = deepcopy(grounded_pred)
        grounded_pred.params = []

        return grounded_pred

class PredicateState:
    def __init__(self, predicates):
        """
        Initializes the predicate state.
        Accepts a list of Predicate objects.
        """
        self.pred_dict = {pred: pred.truth_value for pred in predicates}

    def __eq__(self, other):
        if not isinstance(other, PredicateState):
            return False
        return self.pred_dict == other.pred_dict

    def __hash__(self):
        items = tuple(sorted(self.pred_dict.items(), key=lambda x: hash(x[0])))
        return hash(items)

    def set_pred_value(self, pred_obj, value):
        if pred_obj in self.pred_dict:
            pred_obj.set_truth_value(value)
            self.pred_dict[pred_obj] = value
        else:
            raise Exception("Predicate not found!")

    def get_pred_value(self, pred_obj):
        return self.pred_dict.get(pred_obj, {})

    def add_pred_list(self, new_pred_list):
        """
        Adds new Predicate objects to the state if they don't already exist.
        """
        for pred in new_pred_list:
            if pred not in self.pred_dict:
                self.pred_dict[pred] = pred.truth_value

    def get_unevaluated_predicates(self):
        return [pred for pred, value in self.pred_dict.items() if value is None]

    def iter_predicates(self):
        """
        Generator that yields each predicate object.
        """
        for pred in self.pred_dict:
            yield pred

    def get_pred_list(self, lifted=False):
        """
        Returns a list of predicate dictionaries in original form.
        If lifted=True, params are emptied.
        """
        pred_list = []
        seen = set()
        for pred in self.pred_dict:
            if pred not in seen:
                pred_list.append({
                    "name": pred.name,
                    "types": list(pred.types),
                    "params": [] if lifted else list(pred.params),
                    "semantic": pred.semantic
                })
                seen.add(pred)
        return pred_list

class Precondition:
    def __init__(self, precond: list[Predicate], object_indices: list[list[int]]):
        """
        Representing precondition as a list of predicates with truth values.
        object_indices are indices to map parameters of the operator to each predicate,
        e.g., for skill PlaceAt(object, location), if we have
        op(?obj1 - object, ?loc1 - location, ?obj2 - object, ?obj3 - object)
            precondition:
                (Not(At(obj1, loc1))) and (Not(IsHolding(obj2))) and (Not(At(obj3, loc1)))
        
        precond :: [at: Predicate, isholding: Predicate, at: Predicate]
        object_indices :: [[0, 1], [2], [3, 1]]
        """
        assert len(precond) == len(object_indices), "number of preconditions and object indices must agree"
        self.precond = precond
        self.object_indices = object_indices

    def ground(self, objects: list[str]) -> list[Predicate]:
        """
        Ground the lifted precondition using a list of object names.
        Each object in the list corresponds to a parameter in the operator.
        """
        grounded_precond = []
        for pred, indices in zip(self.precond, self.object_indices):
            params = [objects[i] for i in indices]
            grounded_pred = Predicate(
                name=pred.name,
                types=pred.types,
                params=params,
                semantic=pred.semantic
            )
            grounded_pred.set_truth_value(pred.truth_value)
            grounded_precond.append(grounded_pred)
        return grounded_precond

class Effect:
    def __init__(self, effect_pos: list[Predicate], effect_neg: list[Predicate], object_indices_pos: list[list[str]], object_indices_neg: list[list[str]]):
        assert len(effect_pos) == len(object_indices_pos) and len(effect_neg) == len(object_indices_neg), "number of preconditions and object indices must agree"
        self.effect_pos = effect_pos
        self.effect_neg = effect_neg

        self.object_indices_pos = object_indices_pos
        self.object_indices_neg = object_indices_neg

    def ground(self, objects_pos: list[str], objects_neg: list[str]) -> tuple[list[Predicate]]:
        """
        Ground both eff+ and eff- using separate obejcts list
        """
        grounded_eff_pos = []
        for pred, indices in zip(self.precond, self.object_indices_pos):
            params = [objects_pos[i] for i in indices]
            grounded_pred = Predicate(
                name=pred.name,
                types=pred.types,
                params=params,
                semantic=pred.semantic
            )
            grounded_pred.set_truth_value(pred.truth_value)
            grounded_eff_pos.append(grounded_pred)

        grounded_eff_neg = []
        for pred, indices in zip(self.precond, self.object_indices_neg):
            params = [objects_neg[i] for i in indices]
            grounded_pred = Predicate(
                name=pred.name,
                types=pred.types,
                params=params,
                semantic=pred.semantic
            )
            grounded_pred.set_truth_value(pred.truth_value)
            grounded_eff_neg.append(grounded_pred)

        return (grounded_eff_pos, grounded_eff_neg)

class Operator:
    def __init__(self, skill: Skill, additional_types: list[str], precondition: Precondition, effect: Effect):
        """
        Operator class consists of precondition, effect, and parameters.
        All parameters not in the corresponding skill of the operator will be in additional_types
        """
        self.skill: Skill = skill
        # TODO: automatically calculate the parameter belongs to the skill and the rest
        self.additional_types: list[str] = additional_types

        self.precondition = precondition
        self.effect = effect

    def in_precondition(self, grounded_state: PredicateState, grounded_skill: Skill, type_dict: dict[object: str, list[str]]):
        """
        If the lifted operator can be executed from a grounded state by a grounded skill.
        s \in precond
        """
        skill_params = grounded_skill.params
        # put skill parameters at the begining
        # TODO: this is wrong
        type_list = skill_params + self.additional_types
        possible_groundings = generate_possible_groundings(type_list, type_dict, fixed_grounding=skill_params)
        possible_grounded_predicate_states_precond: list[PredicateState]= [PredicateState(self.precondition.ground(grounding)) for grounding in possible_groundings]
        
        # if there exists one possible grounded predicate state
        # apply all predicates truth value to filter over all possible states
        for grounded_predicate in grounded_state:
            possible_grounded_predicate_states = [predicate_state for predicate_state in possible_grounded_predicate_states_precond \
                                                  if predicate_state[grounded_predicate] == grounded_predicate.get_pred_value(grounded_predicate)]
            if not possible_grounded_predicate_states:
                return False
            
        return True
    
    def in_effect(self, grounded_state: PredicateState, grounded_skill: Skill, type_dict: dict[object: str, list[str]]):
        """
        If the grounded state can be induced by a grounded skill with this operator.
        s \in (precond/eff-) U (eff+)
        """
        skill_params = grounded_skill.params
        type_list = skill_params + self.additional_types
        possible_groundings = generate_possible_groundings(type_list, type_dict, fixed_grounding=skill_params)
        possible_grounded_predicate_states_precond: list[PredicateState]= [PredicateState(self.precondition.ground(grounding)) for grounding in possible_groundings]
        possible_grounded_predicate_states_eff: list[PredicateState]= [PredicateState(self.effect.ground(grounding)) for grounding in possible_groundings]
        for grounded_predicate in grounded_state:

        pass

def generate_possible_groundings(type_list: list[str], type_dict: dict[str, list[str]], fixed_grounding=None) -> list[list[object: str]]:
    """
    required_types: list of types corresponding to total argument slots
    type_dict: dict of object -> type
    fixed_grounding: list of object names fixed from the grounded skill
    """
    if fixed_grounding is None:
        fixed_grounding = []

    # Step 1: Validate fixed_grounding length
    if len(fixed_grounding) > len(type_list):
        raise ValueError("Fixed grounding has more objects than required types.")

    # Step 2: Remove fixed types and objects
    remaining_types = type_list[len(fixed_grounding):]
    used_objects = set(fixed_grounding)

    # Step 3: Invert type_dict to type -> [objects]
    type_to_objects = {}
    for obj, tp in type_dict.items():
        if obj not in used_objects:
            type_to_objects.setdefault(tp, []).append(obj)

    # Step 4: Gather object choices for remaining types
    object_choices = [type_to_objects[tp] for tp in remaining_types]

    # Step 5: Generate combinations and filter duplicates
    combinations = []
    for combo in itertools.product(*object_choices):
        full_combo = tuple(fixed_grounding) + combo
        if len(set(full_combo)) == len(full_combo):
            combinations.append(full_combo)

    return combinations