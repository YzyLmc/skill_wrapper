'Get symbolic representation from skill semantic info and observation'
from utils import GPT4, load_from_file

def predicates_per_skill(prompt, obs, model, skill=None):
    '''
    Propose set of predicates for one skill and observations (optionally)
    '''
    # return model.generate_multimodal(prompt, obs)
    return model.generate(prompt)

def task_proposal(prompt):
    '''
    Proposing task for each predicate of a skill.
    It should be able to prpoposed based on existing roll-outs.
    '''
    pass

def unifiy_predicates():
    '''
    Make sure predicates for different skills have consistent names
    '''
    pass

def predicates2precond():
    '''
    Compose predicates to precondition
    '''
    pass

def predicates2effect():
    '''
    Compose predicates to effect
    '''
    pass

def symbolize():
    '''
    
    '''

if __name__ == "__main__":
    imgs = ["test_imgs/test_0.png"]
    prompt_path = 'prompts/predicates_proposing.txt'
    prompt = load_from_file(prompt_path)
    model = GPT4()
    predicates = predicates_per_skill(prompt, imgs, model)
    print(predicates)

