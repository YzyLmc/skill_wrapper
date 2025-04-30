import itertools
import yaml

class Skill:
    def __init__(self, name, types, params=[], semantics=dict()):
        self.name = name
        self.types = tuple(types)
        self.params = tuple(params)
        self.semantics = semantics
    
    def __str__(self):
        param_str = ", ".join(map(str, self.params))
        type_str = ", ".join(map(str, self.types))
        return f"{self.name}({param_str})" if self.params else f"{self.name}({type_str})"
    
    def __hash__(self):
        return hash((self.name, self.types, self.params))
    
    def __eq__(self, other):
        if not isinstance(other, Skill):
            return False
        return (self.name, self.types, self.params) == (other.name, other.types, other.params)

    def is_grounded(self):
        return bool(self.params)
    
    def ground_with(self, params: list[str], type_dict=None):
        """
        Grounded a skill or a predicate with parameters and their types.

        Args:
            lifted_skill :: Skill object
            params :: list:: list of parameters, e.g., ["Apple", "Table"]
            type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
        """
        assert not self.is_grounded(), "Cannot ground a grounded skill"
        # tuple is applicable to the lifted representation
        if type_dict:
            for i, p in enumerate(params):
                assert p in type_dict
                assert self.types[i] in type_dict[p]
        # grounded skill
        grounded_skill = Skill(
            name=self.name,
            types=self.types,
            params=params
        )
        return grounded_skill
    
    def lifted(self, type_dict=None):
        assert self.is_grounded(), "Cannot lift an ungrounded predicate"
        if type_dict:
            assert all([type in type_dict[param] for type, param in zip(self.types, self.params)])
        
        # lifted skill by remove the parameters
        lifted_skill = Skill(
            name=self.name,
            types=self.types,
            params=[]
        )

        return lifted_skill

class Predicate:
    def __init__(self, name, types, params=[], semantic=None):
        self.name = name
        self.types = tuple(types)
        self.params = tuple(params) if params else ()
        self.semantic = semantic

    def __str__(self):
        param_str = ", ".join(map(str, self.params))
        type_str = ", ".join(map(str, self.types))
        return f"{self.name}({param_str})" if self.params else f"{self.name}({type_str})"

    def __hash__(self):
        # Use name, types, and params as hash — exclude semantic
        return hash((self.name, self.types, self.params))

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return (self.name, self.types, self.params) == (other.name, other.types, other.params)

    def is_grounded(self):
        return bool(self.params)
    
    def ground_with(self, params: list[str], type_dict=None):
        """
        Grounded a skill or a predicate with parameters and their types.
        lifted_pred :: Predicate object
        params :: list:: list of parameters, e.g., ["Apple", "Table"]
        type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
        """
        if self.types: # if the predicate is parametrized
            assert not self.is_grounded(), "Cannot ground an already grounded predicate"
        # tuple is applicable to the lifted representation
        if type_dict:
            for i, p in enumerate(params):
                assert p in type_dict
                assert self.types[i] in type_dict[p]

        # grounded predicate
        grounded_pred = Predicate(
            name=self.name,
            types=self.types,
            params=params,
            semantic=self.semantic
        )

        return grounded_pred

    def lifted(self, type_dict=None):
        if self.types: # if the predicate is parametrized
            assert self.is_grounded(), "Cannot lift an ungrounded predicate"
        if type_dict:
            assert all([type in type_dict[param] for type, param in zip(self.types, self.params)]) or (not self.types)
        
        # lift predicate
        lifted_pred = Predicate(
            name=self.name,
            types=self.types,
            params=[],
            semantic=self.semantic
        )

        return lifted_pred

class PredicateState:
    def __init__(self, predicates):
        """
        Initializes the predicate state.
        Accepts a list of Predicate objects.
        """
        self.pred_dict = {pred: None for pred in predicates}

    def __eq__(self, other):
        if not isinstance(other, PredicateState):
            return False
        return self.pred_dict == other.pred_dict

    def __hash__(self):
        items = tuple(sorted(self.pred_dict.items(), key=lambda x: hash(x[0])))
        return hash(items)

    def __str__(self):
        string = "\n".join([f"{str(pred)} {truth_value}" for pred, truth_value in self.pred_dict.items()])
        return string
    
    def set_pred_value(self, pred_obj, value):
        if pred_obj in self.pred_dict:
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
                self.pred_dict[pred] = None

    def get_unevaluated_preds(self):
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
        pred_list = list(self.pred_dict.keys())
        if lifted:
            pred_list = list(set([pred.lifted() for pred in pred_list]))
        return pred_list

# Customized yaml config
# Save and load data structures
def predicate_representer(dumper, data):
    return dumper.represent_mapping(u'!Predicate', {
        'name': data.name,
        'types': data.types,
        'params': data.params,
        'semantic': data.semantic,
    })

yaml.add_representer(Predicate, predicate_representer)

def predicate_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    pred = Predicate(
        name=values['name'],
        types=values['types'],
        params=values['params'],
        semantic=values['semantic']
    )
    return pred
yaml.add_constructor(u'!Predicate', predicate_constructor)

def predicate_state_representer(dumper, data):
    # Convert Predicate objects and their truth values to a serializable list
    pred_list = []
    for pred, value in data.pred_dict.items():
        pred_list.append({
            'predicate': pred,
            'truth_value': value
        })
    
    return dumper.represent_mapping(u'!PredicateState', {
        'predicates': pred_list
    })
yaml.add_representer(PredicateState, predicate_state_representer)

def predicate_state_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    pred_list = values['predicates']
    
    # Create a new PredicateState from the list of Predicate objects
    preds = [item['predicate'] for item in pred_list]
    state = PredicateState(preds)

    # Set the truth values
    for item in pred_list:
        state.pred_dict[item['predicate']] = item['truth_value']
    
    return state
yaml.add_constructor(u'!PredicateState', predicate_state_constructor)

def skill_representer(dumper, data):
    return dumper.represent_mapping('!Skill', {
        'name': data.name,
        'types': list(data.types),
        'params': list(data.params),
        'semantics': data.semantics
    })
yaml.add_representer(Skill, skill_representer)
yaml.add_representer(Skill, skill_representer, Dumper=yaml.SafeDumper)

def skill_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return Skill(
        name=values["name"],
        types=values["types"],
        params=values["params"],
        semantics=values["semantics"]
    )
yaml.add_constructor('!Skill', skill_constructor)
yaml.add_constructor('!Skill', skill_constructor, Loader=yaml.FullLoader)

if __name__ == "__main__":
    lifted_pred_list = [
        Predicate("At", ["object", "location"]),
        Predicate("CloseTo", ["robot", "location"]),
        Predicate("HandOccupied", []),
        Predicate("IsHolding", ["object"]),
        Predicate("EnoughBattery", []),
        Predicate('handEmpty', [])
    ]
    PickUp = Skill("PickUp", ["object", "location"])
    str2skill = {"pickup": PickUp}
    # Saving to YAML
    with open("str2skill.yaml", "w") as f:
        yaml.dump(str2skill, f)

    # Loading from YAML
    with open("str2skill.yaml", "r") as f:
        loaded_data = yaml.load(f, Loader=yaml.FullLoader)
    breakpoint()