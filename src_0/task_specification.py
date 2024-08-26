from utils import GPT4, load_from_file
import inspect

# Learned predicates
skill2preds = {
    'PickUp(object, location)': ['AtLocation(object,location)', 'Holding(object)', 'At(location)', 'IsReachable(object)', 'IsFreeHand()'],
    'DropAt(object, location)': ['Clear(location)', 'AtLocation(object,location)', 'Holding(object)', 'At(location)', 'IsFreeHand()'],
    'GoTo(location)': ['At(location)', 'IsFreeHand()', 'Clear(location)', 'BatterySufficient()']
               }

all_preds = list(set(sum(skill2preds.values(),[])))
# all_pred = ['Holding(object)', 'At(location)', 'IsReachable(object)', 'Clear(location)', 'IsFreeHand()', 'AtLocation(object,location)', 'BatterySufficient()']
# While 'Clear(location) 'BatterySufficient()' 'IsReachable(object)' are discarded or handled internally
skills = {
    "PickUp(object, location)": 
    {
        "precond": ["AtLocation(object,location)", "At(location)", "IsReachable(object)", "IsFreeHand()"],
        "eff+": ["Holding(object)"],
        "eff-": ["AtLocation(object,location)", "IsFreeHand()"]
    }
    ,
    "DropAt(object, location)":{
        "precond": ["At(location)", "Holding(object)"],
        "eff+": ["AtLocation(object,location)", "IsFreeHand()"],
        "eff-": ["Holding(object)"]
    }
    ,
    "GoTo(location)":{
        "precond": [],
        "eff+": ["At(location)"],
        "eff-": []
    }
}
