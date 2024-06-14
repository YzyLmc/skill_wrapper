'adapt original skill into current environment by combining it with lower level actions'

def get_nearest_pair(observation, database):
    '''get a contrastive pair from the database'''
    pass

def predict_success(skill, observation, pair):
    '''
    predict if the execution will succeed.
    Multiple backends including GPT, cosine similarity.
    '''
    pass

def back_to_success_state(observation, success_state, prompt):
    '''
    Drive back the agent to the successful state. Only triggered when predicted to fail
    '''
    pass

def adapt(observation, database, prompt):
    '''
    Main function for adaptation
    '''
    pass