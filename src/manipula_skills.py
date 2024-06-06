'''wrapped skills for manipulaThor agent desgined for the adaptation pipeline, both high-level and low-level'''

# Base movement
def MoveForward(controller):
    pass

def MoveBackward(controller):
    pass

def MoveLeft(controller):
    pass

def MoveRight(controller):
    pass

def GoTo(controller, obj_name):
    '''Teleport to a distance from the obj *The goto function has to be imperfect*'''
    pass

def LookAt(controller, obj_name):
    '''rotate the base to center the obj. No idea how to look up or down for centering.'''
    pass

# Gripper movement
def GripperUp(controller):
    pass

def GripperDown(controller):
    pass

def GripperForward(controller):
    pass

def GripperBackward(controller):
    pass

def PickUp(controller, obj_name):
    pass

def Drop(controller):
    pass
