# from src.data_structures.ParameterizedLiftedRelation import ParameterizedLiftedRelation
# from src.data_structures.Parameter import Parameter
# from src.data_structures.Link import Link
# from src.data_structures.PDDLPrecondition import LiftedPDDLPrecondition
# from src.data_structures.PDDLEffect import LiftedPDDLEffect
import copy
from copy import deepcopy
# import Config
import numpy as np
from itertools import product
# from src.useful_functions import print_set
import functools
# from openravepy.misc import DrawAxes

@functools.total_ordering
class Link(object):
    def __init__(self, link_name, link_type):
        self.name = link_name
        self.type = link_type
    
    def __eq__(self,o):
        if type(o) == type("s"): 
            if self.name == o: 
                return True
            else:
                return False
        if self.name == o.name and self.type == o.type:
            return True
        else:
            return False
    
    def __lt__(self,o):
        if self.name < o.name:
            return True
        else:
            return False
    
    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.__str__())

@functools.total_ordering
class Parameter(object):
    def __init__(self,pid, type, name=None):
        self.name = name
        self.pid = pid
        self.type = type

    def __str__(self):
        return str(self.pid)
    
    def __hash__(self):
        return hash(self.__str__())
    
    def __eq__(self,o):
        if self.type == o.type and self.pid == o.pid:
            return True
        else:
            return False

    def __lt__(self,o):
        if self.type < o.type:
            return True
        elif self.type == o.type and self.pid < o.pid:
            return True
        else:
            return False

    def get_grounded_parameter(self,value):
        return GroundedParameter(self.pid, self.type, value)

class GroundedParameter(Parameter):
    def __init__(self,pid, type, value):
        super(GroundedParameter,self).__init__(pid,type)
        self.value = value

    def __str__(self):
        return super(GroundedParameter,self).__str__() + " : " + str(self.value)
    
    def hash(self):
        return hash(self.__str__())
    
    def __eq__(self,o):
        if super(GroundedParameter,self).__eq__(o) and self.value == o.value:
            return True
        else:
            return False

@functools.total_ordering
class ParameterizedLiftedRelation(object):
    def __init__(self, pid1, pid2, parent_relation): 
        self.pid1 = pid1
        self.pid2 = pid2 
        self.parent_relation = parent_relation

    def ground_relation(self,grounding): 
        if "Const" in self.pid1:
            grounded_param1 = Link(link_name=self.pid1,link_type=self.pid1.split("_")[0])
        else:
            grounded_param1 = grounding[self.pid1]
            
        if "Const" in self.pid2:
            grounded_param2 = Link(link_name=self.pid2,link_type=self.pid2.split("_")[0])
        else:
            grounded_param2 = grounding[self.pid2]

        return self.parent_relation.get_grounded_relation(grounded_param1, grounded_param2)    

    def __str__(self):
        # if self.parent_relation.parameter1_type in Config.CONST_TYPES[Config.DOMAIN_NAME]:
        #     return "({}_{}_{} {} ?{})".format(self.parent_relation.parameter1_type, self.parent_relation.parameter2_type,self.parent_relation.cr, self.pid1, self.pid2)
        # elif self.parent_relation.parameter2_type in Config.CONST_TYPES[Config.DOMAIN_NAME]:
        #     return "({}_{}_{} ?{} {})".format(self.parent_relation.parameter1_type, self.parent_relation.parameter2_type,self.parent_relation.cr, self.pid1, self.pid2)
        
        return "({}_{}_{} ?{} ?{})".format(self.parent_relation.parameter1_type, self.parent_relation.parameter2_type,self.parent_relation.cr, self.pid1, self.pid2)
    
    def __hash__(self):
        return hash(self.__str__())
    
    def __eq__(self,o):
        if self.pid1 != o.pid1:
            return False
        if self.pid2 != o.pid2:
            return False
        if self.parent_relation != o.parent_relation:
            return False

        return True
    
    def __lt__(self,o):
        return self.__str__() < o.__str__()

class GroundedPDDLPrecondition(object): 
    def __init__(self,true_set,false_set): 
        self.true_set = true_set
        self.false_set = false_set
    
    def check_in_state(self,pddl_state):
        for prop in self.true_set: 
            if prop not in pddl_state.true_set:
                return False
        for prop in self.false_set:
            if prop in pddl_state.true_set: 
                return False
        return True

@functools.total_ordering
class LiftedPDDLPrecondition(object):
    def __init__(self, true_set, false_set,true_aux_set,false_aux_set=set()):
        self.true_set = true_set
        self.false_set = false_set
        self.true_aux_set = true_aux_set
        self.false_aux_set = false_aux_set

    def get_grounded_precondition(self,grounding): 
        true_set = set()
        false_set = set() 
        for prop in self.true_set: 
            true_set.add(prop.ground_relation(grounding))
        for prop in self.false_set: 
            false_set.add(prop.ground_relation(grounding))
        return GroundedPDDLPrecondition(true_set, false_set)
    
    def get_lifted_true_set(self):
        lifted_set = set([])
        for param_re in self.true_set:
            lifted_set.add(param_re.parent_relation)
        
        return lifted_set
    
    def get_lifted_false_set(self):
        lifted_set = set([])
        for param_re in self.false_set:
            lifted_set.add(param_re.parent_relation)
        
        return lifted_set

    def sort_set(self,to_sort):
        sort_list = list(to_sort)
        sort_list.sort()

        return sort_list
    
    def __eq__(self,o):
        if (self.true_set != o.true_set) or (self.false_set != o.false_set) or (self.true_aux_set != o.true_aux_set) or (self.false_aux_set != o.false_aux_set):
            return False
        return True
    
    def __lt__(self,o):
        return self.__str__() < o.__str__()
    
    def __str__(self):
        precondition_string = ""
        for prop in self.sort_set(self.true_set):
            precondition_string += "\t{}\n".format(str(prop))
        
        for prop in self.sort_set(self.false_set):
            precondition_string += "\t(not {})\n".format(str(prop))

        auxillary_string = ""
        for a_prop in self.sort_set(self.true_aux_set):
            if a_prop.id <= 2:
                auxillary_string += "\t({}) \n".format(str(a_prop))
            else:
                # if str(a_prop).split()[1].split("_")[0] in Config.CONST_TYPES[Config.DOMAIN_NAME]:
                #     s_ap = str(a_prop).split()[0] + " " + str(a_prop).split()[1]
                # else:
                s_ap = str(a_prop).split()[0] + " ?" + str(a_prop).split()[1]

                auxillary_string += "\t({}) \n".format(s_ap)

        for a_prop in self.sort_set(self.false_aux_set):
            if a_prop.id <= 2:
                auxillary_string += "\t({}) \n".format(str(a_prop))
            else:
                # if str(a_prop).split()[1].split("_")[0] in Config.CONST_TYPES[Config.DOMAIN_NAME]:
                #     s_ap = str(a_prop).split()[0] + " " + str(a_prop).split()[1]
                # else:
                s_ap = str(a_prop).split()[0] + " ?" + str(a_prop).split()[1]

                auxillary_string += "\t(not ({})) \n".format(s_ap)
        
        precondition_string+=auxillary_string

        return precondition_string

    def __hash__(self):
        return hash(self.__str__())
    
class GroundedPDDLEffect(object):
    def __init__(self, add_set, delete_set): 
        self.add_set = add_set
        self.delete_set = delete_set 

    def apply(self, pddl_state):
        new_state = copy.deepcopy(pddl_state)
        for prop in self.delete_set: 
            new_state.true_set.remove(prop)
        for prop in self.add_set:
            new_state.true_set.add(prop)
        return new_state

@functools.total_ordering
class LiftedPDDLEffect(object):
    def __init__(self, add_set, delete_set,aux_add, aux_delete): 
        self.add_set = add_set 
        self.delete_set = delete_set
        self.aux_add = aux_add
        self.aux_delete = aux_delete

    def get_grounded_effect(self,grounding): 
        add_set = set()
        delete_set = set() 
        for prop in self.add_set: 
            add_set.add(prop.ground_relation(grounding))
        for prop in self.delete_set: 
            delete_set.add(prop.ground_relation(grounding))
        return GroundedPDDLEffect(add_set, delete_set)

    def get_lifted_add_set(self):
        lifted_set = set([])
        for param_re in self.add_set:
            lifted_set.add(param_re.parent_relation)
        
        return lifted_set
    
    def get_lifted_delete_set(self):
        lifted_set = set([])
        for param_re in self.delete_set:
            lifted_set.add(param_re.parent_relation)
        
        return lifted_set
    
    def sort_set(self,to_sort):
        sort_list = list(to_sort)
        sort_list.sort()

        return sort_list
    
    def __eq__(self,o):
        if (self.add_set != o.add_set) or (self.delete_set != o.delete_set) or (self.aux_add != o.aux_add) or (self.aux_delete != o.aux_delete):
            return False
        return True
    
    def __lt__(self,o):
        return self.__str__() < o.__str__()
    
    def __str__(self):
        effect_string = ""
        for prop in self.sort_set(self.add_set):
            effect_string += "\t{} \n".format(str(prop))
        for prop in self.sort_set(self.delete_set):
            effect_string += "\t(not {})\n".format(str(prop))
        
        auxillary_string = ""
        for a_prop in self.sort_set(self.aux_add):
            if a_prop.id <= 2:
                auxillary_string += "\t({}) \n".format(str(a_prop))
            else:
                # if str(a_prop).split()[1].split("_")[0] in Config.CONST_TYPES[Config.DOMAIN_NAME]:
                #     s_ap = str(a_prop).split()[0] + " " + str(a_prop).split()[1]
                # else:
                s_ap = str(a_prop).split()[0] + " ?" + str(a_prop).split()[1]

                auxillary_string += "\t({}) \n".format(s_ap)

        for a_prop in self.sort_set(self.aux_delete):
            if a_prop.id <= 2:
                auxillary_string += "\t(not ({})) \n".format(str(a_prop))
            else:
                # if str(a_prop).split()[1].split("_")[0] in Config.CONST_TYPES[Config.DOMAIN_NAME]:
                #     s_ap = str(a_prop).split()[0] + " " + str(a_prop).split()[1]
                # else:
                s_ap = str(a_prop).split()[0] + " ?" + str(a_prop).split()[1]

                auxillary_string += "\t(not ({})) \n".format(s_ap)
        
        effect_string+=auxillary_string

        return effect_string

    def __hash__(self):
        return hash(self.__str__())

@functools.total_ordering
class PDDLState(object):
    def __init__(self,true_set,false_set,aux_true_set = set(),aux_false_set = set()): 
        self.true_set = true_set
        self.false_set = false_set
        self.aux_true_set = aux_true_set
        self.aux_false_set = aux_false_set

    def is_relation_true(self,grounded_relation): 
        if grounded_relation in self.true_set: 
            return True
        else:
            return False
    
    # @staticmethod
    # def get_from_ll(lifted_relations_dict, object_dict, ll_state, aux_list):
        true_set = set()
        false_set = set()
        aux_true_set = set()

        for object_pair in lifted_relations_dict: 
            for cr in lifted_relations_dict[object_pair]: 
                relation = lifted_relations_dict[object_pair][cr]
                l1 = object_dict[relation.parameter1_type]
                l2 = object_dict[relation.parameter2_type]
                combinations = product(l1,l2)
                for combination in combinations:
                    grounded_relation = relation.get_grounded_relation(combination[0],combination[1])
                    if grounded_relation.evaluate_in_ll_state(ll_state): 
                        true_set.add(grounded_relation)
                    else:
                        false_set.add(grounded_relation)

        for relation in aux_list:
            if relation.id == 1:
                flag = True
                for r in true_set:
                    if r.parameter1_type == relation.parameter1_type and r.parameter2_type == relation.parameter2_type and relation.cr == r.cr:
                        flag = False
                        break
                if flag:
                    aux_true_set.add(relation)

            elif relation.id == 2:
                n1 = len(object_dict[relation.parameter1_type])
                n2 = len(object_dict[relation.parameter2_type])
                c = 0
                for r in true_set:
                    if r.parameter1_type == relation.parameter1_type and r.parameter2_type == relation.parameter2_type and relation.cr == r.cr:
                        c += 1
                if c == n1 * n2:
                    aux_true_set.add(relation)

            elif relation.id == 3:
                flag = True                
                for r in true_set:
                    if (r.parameter1_type == relation.parameter1_type) and (r.parameter2_type == relation.parameter2_type) and (r.cr == relation.cr) and (r.parameter1 == relation.parameter):
                        flag = False
                        break
                if flag:
                    aux_true_set.add(relation)
            
            elif relation.id == 4:
                flag = True
                for r in  true_set:
                    if (r.parameter1_type == relation.parameter1_type) and (r.parameter2_type == relation.parameter2_type) and (r.cr == relation.cr) and (r.parameter2 == relation.parameter):
                        flag = False
                        break
                if flag:
                    aux_true_set.add(relation)
                    
        return PDDLState(true_set,false_set,aux_true_set,set())
    
    def __str__(self):
        s = ""
        true_set_list = list(self.true_set)
        true_set_list.sort()
        for i,prop in enumerate(true_set_list): 
            s += str(prop)
            s += " "
            if i > 0 and i % 4 == 0: 
                s += "\n"
        
        aux_list = list(self.aux_true_set)
        aux_list.sort()
        for i, ap in enumerate(aux_list):
            s += "("+str(ap)+")"
            s += " "
            if i > 0 and i % 4 == 0: 
                s += "\n"
        return s

    def __eq__(self,o):
        if len(self.true_set) != len(o.true_set):
            return False
        if len(self.aux_true_set) != len(o.aux_true_set):
            return False

        for prop in self.true_set: 
            if prop not in o.true_set: 
                return False        
        for aux_prop in self.aux_true_set:
            if aux_prop not in o.aux_true_set:
                return False

        return True
    
    def __lt__(self,o):
        return True
    
    def __hash__(self):
        return hash(self.__str__())

    def __deepcopy__(self,memodict={}):
        new_pddl_state = PDDLState(self.true_set,self.false_set,self.aux_true_set,self.aux_false_set)
        return new_pddl_state

# TODO: make it compatible with 0 or 1 parameters  
@functools.total_ordering
class Relation(object):    
    def __init__(self, parameter1_type, parameter2_type, cr, region=None, discretizer=None): 
        self.parameter1_type = parameter1_type
        self.parameter2_type = parameter2_type
        self.cr = cr
        self.region = region
        self.discretizer = discretizer

    def get_grounded_relation(self,parameter1, parameter2): 
        return GroundedRelation(parameter1,parameter2,self.cr,self.region,self.discretizer)
    
    def __str__(self): 
        return "({}_{}_{} ?x - {} ?y - {})".format(self.parameter1_type,self.parameter2_type, str(self.cr), self.parameter1_type, self.parameter2_type )

    def __eq__(self,o): 
        if self.parameter1_type == o.parameter1_type and self.parameter2_type == o.parameter2_type and self.region == o.region:
            if self.cr == 0 or o.cr == 0:
                return self.cr == o.cr
            return True
        elif self.parameter1_type == o.parameter2_type and self.parameter2_type == o.parameter1_type and self.region == o.region:
            if self.cr == 0 or o.cr == 0:
                return self.cr == o.cr
            return True
        else: 
            return False
        
    def __lt__(self,o): 
        return self.__str__() < o.__str__()
        
    def __hash__(self):
        # return hash(self.__str__())
        if self.cr != 0:
            return hash("({}_{}_{} ?x - {} ?y - {})".format(self.parameter1_type,self.parameter2_type, str(self.region), self.parameter1_type, self.parameter2_type ))
        else:
            return hash("({}_{}_{} ?x - {} ?y - {})".format(self.parameter1_type,self.parameter2_type, str(len(self.region)), self.parameter1_type, self.parameter2_type ))

    def __deepcopy__(self,memodict={}):
        region_to_copy = copy.deepcopy(self.region)
        new_relation = Relation(self.parameter1_type, self.parameter2_type, self.cr, region_to_copy, self.discretizer)
        return new_relation
            
@functools.total_ordering
class GroundedRelation(Relation): 
    def __init__(self,parameter1,parameter2,cr,region=None,discretizer=None):
        super(GroundedRelation,self).__init__(parameter1.type, parameter2.type,cr,region,discretizer)
        self.p1 = parameter1
        self.p2 = parameter2
        self.parameter1 = parameter1.name
        self.parameter2 = parameter2.name
        self.relational = True if parameter1.type != "world" else False
        self.region_generator = None 
        self.sample_fn = None 
        self.env_state = None 
        self.sim_object = None 
        self.region_to_use  = None 

    def __deepcopy__(self, memodict={}): 
        region_to_copy = copy.deepcopy(self.region)
        new_relation = GroundedRelation(self.p1, self.p2, self.cr,region_to_copy,self.discretizer)
        return new_relation

    def __str__(self):
        parameter1_str = self.parameter1
        # if self.parameter1_type in Config.CONST_TYPES[Config.DOMAIN_NAME]:
        #     parameter1_str = self.parameter1_type + "_Const"
        
        parameter2_str = self.parameter2
        # if self.parameter2_type in Config.CONST_TYPES[Config.DOMAIN_NAME]:
        #     parameter2_str = self.parameter2_type + "_Const"
    
        return "({}_{}_{} {} {})".format(self.parameter1_type, self.parameter2_type, str(self.cr), parameter1_str, parameter2_str) 
    
    def evaluate(self, state): 
        s = self.__str__()
        if s in state: 
            return True
        else:
            return False
    
    def __eq__(self,o):
        if not super(GroundedRelation,self).__eq__(o):
            return False 
        elif self.parameter1 == o.parameter1 and self.parameter2 == o.parameter2 and self.region == o.region:
            return True
        elif self.parameter1 == o.parameter2 and self.parameter2 == o.parameter1 and self.region == o.region:
            return True
        else: 
            return False
        
    def __lt__(self,o): 
        return super(GroundedRelation,self).__lt__(o)
        
    def __hash__(self):
        # return hash(self.__str__())
        parameter1_str = self.parameter1
        # if self.parameter1_type in Config.CONST_TYPES[Config.DOMAIN_NAME]:
        #     parameter1_str = self.parameter1_type + "_Const"
        
        parameter2_str = self.parameter2
        # if self.parameter2_type in Config.CONST_TYPES[Config.DOMAIN_NAME]:
        #     parameter2_str = self.parameter2_type + "_Const"

        if self.cr != 0:
            return hash("({}_{}_{} {} {})".format(self.parameter1_type, self.parameter2_type, str(self.region), parameter1_str, parameter2_str))        
        else:
            return hash("({}_{}_{} {} {})".format(self.parameter1_type, self.parameter2_type, str(len(self.region)), parameter1_str, parameter2_str))        

    def get_lifted_relation(self):
        return Relation(self.parameter1_type,self.parameter2_type,self.cr, self.region,self.discretizer)

    # def evaluate_in_ll_state(self,ll_state):
        link1_relative_pose, link2_relative_pose = self.get_relative_pose(ll_state)
        link1_relative_discretized_pose = self.discretizer.get_discretized_pose(link1_relative_pose,is_relative = self.relational)
        link2_relative_discretized_pose = self.discretizer.get_discretized_pose(link2_relative_pose,is_relative = self.relational)
        grab_flag = 0
        grabbed = False
        for n in range(1,ll_state.num_robots+1):
            grabbed = (grabbed or getattr(ll_state,"grabbed_flag_{}".format(n)))

        if grabbed:
            grabbed_object_flag = False
            for r in range(1,ll_state.num_robots+1):
                if (self.parameter1 == getattr(ll_state,"grabbed_object_{}".format(r)) or self.parameter2 == getattr(ll_state,"grabbed_object_{}".format(r))):
                    grabbed_object_flag = True

            if (self.parameter1_type == Config.GRIPPER_NAME or self.parameter2_type == Config.GRIPPER_NAME) and grabbed_object_flag:
                grab_flag = 0
                if (self.parameter1_type == Config.GRIPPER_NAME) and (self.parameter2_type in Config.OBJECT_NAME):
                    id = self.parameter1.split("_")[1]

                    if grabbed:
                        if self.parameter2 == getattr(ll_state,"grabbed_object_{}".format(id)):
                            grab_flag = 1
                        else:
                            grab_flag = 2
                    else:
                        grab_flag = 0

                elif (self.parameter2_type == Config.GRIPPER_NAME) and (self.parameter1_type in Config.OBJECT_NAME):
                    id = self.parameter2.split("_")[1]
                    
                    if grabbed:
                        if self.parameter1 == getattr(ll_state,"grabbed_object_{}".format(id)):
                            grab_flag = 1
                        else:
                            grab_flag = 2
                    else:
                        grab_flag = 0                  

        link1_relative_discretized_pose.append(grab_flag)
        link2_relative_discretized_pose.append(grab_flag)
        if self.cr == 0: 
            if self.parameter1_type == self.parameter2_type:
                return not (link2_relative_discretized_pose in self.region) #or (link1_relative_discretized_pose in self.region)
            else:
                return not (link1_relative_discretized_pose in self.region or link2_relative_discretized_pose in self.region)
        else:
            if self.parameter1_type == self.parameter2_type:
                return (link2_relative_discretized_pose in self.region) #or (link1_relative_discretized_pose in self.region)
            else:
                return (link1_relative_discretized_pose in self.region or link2_relative_discretized_pose in self.region)

    # def get_grounded_pose(self,lifted_transform, env_state,switch=False):
        object_dic = env_state.object_dict
        if switch:
            object_name = self.parameter2
            if "Const" in self.parameter2:
                object_name = self.parameter2_type + "_" + self.parameter1.split("_")[-1]

        else:
            object_name = self.parameter1
            if "Const" in self.parameter1:
                object_name = self.parameter1_type + "_" + self.parameter2.split("_")[-1]
        
        current_link1_pose = object_dic[object_name]
        if object_name.split("_")[0] in Config.ROBOT_TYPES.keys():
            current_link1_pose = object_dic[object_name][1]
        current_link1_transform = transform_from_pose(current_link1_pose)
        return current_link1_transform.dot(lifted_transform)
    
    # def get_relative_pose(self, env_state):
        object_dic = env_state.object_dict
        parameter1 = self.parameter1
        parameter2 = self.parameter2

        if self.parameter1_type in Config.CONST_TYPES[Config.DOMAIN_NAME]:
            parameter1 = deepcopy(self.parameter1_type + "_" + parameter2.split("_")[-1])

        if self.parameter2_type in Config.CONST_TYPES[Config.DOMAIN_NAME]:
            parameter2 = deepcopy(self.parameter2_type + "_" + parameter1.split("_")[-1])

        if self.parameter1_type in Config.ROBOT_TYPES.keys():
            link1_pose = object_dic[parameter1][1]
        else:
            link1_pose = object_dic[parameter1]

        if self.parameter2_type in Config.ROBOT_TYPES.keys():
            link2_pose = object_dic[parameter2][1]
        else:
            link2_pose = object_dic[parameter2]

        relative_pose_1 = env_state.get_relative_pose(link2_pose,link1_pose)
        relative_pose_2 = env_state.get_relative_pose(link1_pose,link2_pose)
        return relative_pose_1,relative_pose_2

    def get_next_region(self):
        if self.cr!= 0: 
            for region in self.region: 
                yield region 
        else:
            yield self.region[0]

    def sample_region(self): 
        if self.region_generator is None: 
            self.region_generator = self.get_next_region() 
        try: 
            region = self.region_generator.next() 
        except StopIteration:
            self.region_generator = None  
            raise StopIteration
        else: 
            return region 
        
    def init_sample_generator(self,env_state,sim_object, region,action_info):
        samples = [] 
        self.env_state = env_state
        self.sim_object = sim_object
        self.region_to_use = region
        self.sample_fn = self.sampler(action_info)
    
    # def sampler(self,action_info):
        # if self.cr == 0:
        #     n = 1
        # else:
        #     n = Config.SAMPLE_COUNT

        # for i in range(n):
        #     yield self.sample_config(action_info)
    
    # def get_next_sample(self):
        if self.sample_fn is None: 
            raise StopIteration
        try: 
            sample = self.sample_fn.next() 
        except StopIteration:
            self.sample_fn = None 
            raise StopIteration
        else:
            return sample 

    # def sample_config(self,action_info):
        env_state = self.env_state 
        sim_object = self.sim_object 
        region = self.region_to_use
        switch = False
        object_with_transform = self.parameter2
        static_object = self.parameter1
        static_param_num = 1
        obj_list = []

        action_type, action_axis, action_order = action_info

        if self.switch_check(object_with_transform=object_with_transform,env_state=env_state):
            object_with_transform = self.parameter1
            static_object = self.parameter2
            static_param_num = 2
            switch = True

        static_list = [static_object,static_param_num]

        t_robot=None
        for obj in env_state.object_dict.keys():
            if (self.parameter1_type == Config.BASE_NAME or self.parameter2_type == Config.BASE_NAME) and (self.parameter1_type != Config.GRIPPER_NAME and self.parameter2_type != Config.GRIPPER_NAME):
                if Config.BASE_NAME in obj:
                    t_robot = env_state.transform_from_pose(env_state.object_dict[obj][1])
                    robot_type = Config.BASE_NAME          
                    rob_id = obj.split("_")[1]
                    
            if obj == object_with_transform:
                pose = env_state.object_dict[obj]
                if object_with_transform.split("_")[0] in Config.ROBOT_TYPES.keys():
                    pose = env_state.object_dict[obj][1]
                t_obj = env_state.transform_from_pose(pose)
        
        if t_robot is None:
            if Config.GRIPPER_NAME in object_with_transform:
                rob_id = int(object_with_transform.split("_")[1])
            
            else:
                obj = object_with_transform
                for r in range(1,env_state.num_robots + 1):
                    if obj == getattr(env_state,"grabbed_object_{}".format(r)):
                        rob_id = r
                        break                                      
        
            for obj in env_state.object_dict.keys():
                if Config.GRIPPER_NAME in obj and int(obj.split("_")[1]) == rob_id:
                    t_robot = env_state.transform_from_pose(env_state.object_dict[obj][1])
                    break
            
            robot_type = Config.GRIPPER_NAME
        
        for rob in sim_object.robots:
            if rob.id == rob_id:
                break

        if self.cr == 0:
            current_dof = env_state.object_dict[robot_type+"_{}".format(rob.id)][0]
            sampled_config,sampled_end_effector_transform = sim_object.random_config_robot(robot=rob,current_dof=current_dof,exp=True) #TODO: set "exp" to False again
            sampled_config.append(0)
            sampled_lifted_region = np.zeros(shape=Config.BIN_COUNT.shape[0]+1)
            sampled_refined_grounded_region = sampled_config[:-1]
            object_with_transform = robot_type+"_{}".format(rob.id)
            delta_mp = None
            # robot = rob
        
        else:
            sampled_lifted_region = region 

            grab_flag = sampled_lifted_region[-1]
            if ( not switch and "Loc" in self.parameter1 ) or (switch and "Loc" in self.parameter2): 
                sampled_refined_lifted_region = np.eye(4)
            else: 
                sampled_refined_lifted_region = transform_from_pose(self.discretizer.convert_sample(sampled_lifted_region[:6], is_relative = True))

            if switch:
                sampled_refined_lifted_region = np.linalg.pinv(sampled_refined_lifted_region)

            sampled_refined_grounded_region = self.get_grounded_pose(sampled_refined_lifted_region,env_state,switch=switch)

            relative_t = np.linalg.pinv(t_obj).dot(t_robot)
            sampled_end_effector_transform = sampled_refined_grounded_region.dot(relative_t)
            sampled_config = []
            ik_count = 0

            while ik_count < Config.MAX_IK_ATTEMPTS and len(sampled_config) == 0:
                sampled_config = rob.get_ik_solutions(sampled_end_effector_transform,robot_param=object_with_transform.split("_")[0],collision_fn = sim_object.collision_check)
                ik_count += 1

            delta_mp = None
            obj_list = []
            
            sampled_config = list(sampled_config)
            sampled_config.append(grab_flag)

        return sampled_config,sampled_lifted_region,object_with_transform,sampled_refined_grounded_region,rob, static_list, sampled_end_effector_transform, obj_list, delta_mp
        
    # def switch_check(self,object_with_transform,env_state):
        if (object_with_transform.split("_")[0] in Config.IMMOVABLE_OBJECTS):
            return True
        else:
            if ((self.parameter1_type == self.parameter2_type) and (self.parameter1_type in Config.ROBOT_TYPES)):
                return False
            else:
                grabbed_object_flag = False
                for r in range(1,env_state.num_robots+1):
                    if getattr(env_state,"grabbed_object_{}".format(r)) in [self.parameter1, self.parameter2]:
                        grabbed_object_flag = True
                        grabbed_object = getattr(env_state,"grabbed_object_{}".format(r))
                        break

                if grabbed_object_flag:
                    if Config.GRIPPER_NAME in self.parameter1_type:
                        id = self.parameter1.split("_")[1]
                        if object_with_transform == getattr(env_state,"grabbed_object_{}".format(id)):
                            return True
                        else:
                            return False                    
                    elif grabbed_object == self.parameter1:
                        return True
                    else:
                        return False

                else:
                    if self.parameter1_type in Config.ROBOT_TYPES.keys() and self.parameter2_type not in Config.ROBOT_TYPES.keys():
                        return True
                    else:
                        return False
                    
@functools.total_ordering
class LiftedPDDLAction(object): 
    action_id  = 0
    def __init__(self,id, parameters, preconditions, effects, required_planks=set([]),states_to_neglect=set([])):
        self.action_id = id
        self.parameters = parameters
        self.preconditions = preconditions 
        self.effects = effects 
        self.required_planks = required_planks
        self.states_to_neglect = states_to_neglect

    # @staticmethod
    # def get_robot_id_set(relation):
    #     id_set = set([])
    #     if relation.parameter1_type in Config.ROBOT_TYPES:
    #         id_set.add(int(relation.parameter1.split("_")[1]))

    #     if relation.parameter2_type in Config.ROBOT_TYPES:
    #         id_set.add(int(relation.parameter2.split("_")[1]))

    #     return id_set
    
    @staticmethod
    def get_param_objects(param_objects_set,additional_param_objects_dict):
        param_objects = copy.deepcopy(param_objects_set)
        for obj_type in additional_param_objects_dict.keys():
            param_objects = param_objects.union(set(additional_param_objects_dict[obj_type]))

        return param_objects

    @staticmethod
    def get_action_from_cluster(cluster):

        # cluster: List[PDDLGroundedState], List[PDDLGroundedState]


        cluster_e_add = set()
        cluster_e_delete = set()
        changed_relations = set()



        temp_added = set()
        temp_deleted = set()

        for r1 in cluster[0][0].true_set: 
            if r1 not in cluster[0][1].true_set:
                changed_relations.add(r1)
                temp_deleted.add(r1)
        for r1 in cluster[0][1].true_set: 
            if r1 not in cluster[0][0].true_set: 
                changed_relations.add(r1)
                temp_added.add(r1)

        param_ids = {}
        param_mapping = { }
        relation_param_mapping = {}

        for relation in changed_relations: 
            if relation.parameter1_type not in param_ids: 
                param_ids[relation.parameter1_type] = 1
            if relation.parameter2_type not in param_ids: 
                param_ids[relation.parameter2_type] = 1
            if relation.parameter1 not in param_mapping:
                pid1 = param_ids[relation.parameter1_type]
                param_ids[relation.parameter1_type] += 1
                param_mapping[relation.parameter1] = relation.parameter1_type + "_p" + str(pid1)
            if relation.parameter2 not in param_mapping:
                pid2 = param_ids[relation.parameter2_type]
                param_ids[relation.parameter2_type] += 1
                param_mapping[relation.parameter2] = relation.parameter2_type + "_p" + str(pid2)
            lr = relation.get_lifted_relation()
            if lr not in relation_param_mapping:
                relation_param_mapping[lr] = [ [param_mapping[relation.parameter1], param_mapping[relation.parameter2]] ]
            else: 
                relation_param_mapping[lr].append([param_mapping[relation.parameter1], param_mapping[relation.parameter2]])

        # robot_id_set = set([])
        # for relation in cluster[0][0].true_set:
        #     if relation in changed_relations:
        #         id_set = LiftedPDDLAction.get_robot_id_set(relation)
        #         if len(id_set) > 0:
        #             robot_id_set = robot_id_set.union(id_set)
            
        for relation in temp_added:
            lr = relation.get_lifted_relation()
            pid1 = param_mapping[relation.parameter1]
            pid2 = param_mapping[relation.parameter2]
            cluster_e_add.add(ParameterizedLiftedRelation(pid1,pid2,lr))
        
        for relation in temp_deleted:
            lr = relation.get_lifted_relation()
            pid1 = param_mapping[relation.parameter1]
            pid2 = param_mapping[relation.parameter2]
            cluster_e_delete.add(ParameterizedLiftedRelation(pid1,pid2,lr))

        relations_union = cluster[0][0].true_set.union(cluster[0][0].false_set)
        for relation in temp_added:
            p1 = relation.parameter1
            p2 = relation.parameter2
            for p in relations_union:
                if p.parameter1 == p1 and p.parameter2 == p2 and relation != p:
                    pa = param_mapping[p.parameter1]
                    pb = param_mapping[p.parameter2]
                    lifted_relation = p.get_lifted_relation()
                    parameteirzed_relation = ParameterizedLiftedRelation(pa,pb,lifted_relation)
                    cluster_e_delete.add(parameterized_relation)
        

        common_relations = set()
        additional_param_mappings = { }
        additional_param_ids = { }
        extra_robot_id_set = set()        
        param_objects = set([])

        additional_param_objects = {}
        sorted_true_set = list(cluster[0][0].true_set)
        sorted_true_set.sort()
        sorted_true_set = sorted_true_set[::-1]

        for relation in sorted_true_set: 
            lr = relation.get_lifted_relation()
            if relation in changed_relations:
                if len(relation_param_mapping[lr]) == 1:
                    lr_index = 0
                else: 
                    lr_index = -1 
                    for lr_i in range(len(relation_param_mapping[lr])):
                        ps = relation_param_mapping[lr][lr_i]
                        if ps[0] == param_mapping[relation.parameter1] and ps[1] == param_mapping[relation.parameter2]: 
                            lr_index = lr_i 
                            break 
                    if lr_index == -1: 
                        print("It should never come here..")
                        print("something is wrong!!")
                        exit (-1) 
                pid1 = copy.deepcopy(relation_param_mapping[lr][lr_index][0])
                pid2 = copy.deepcopy(relation_param_mapping[lr][lr_index][1])
                parameterized_relation = ParameterizedLiftedRelation(pid1,pid2,lr)
                common_relations.add(parameterized_relation)
        
        for relation in sorted_true_set: 
            lr = relation.get_lifted_relation()
            if relation not in changed_relations:
                # if (((relation.parameter1 in param_mapping and relation.parameter1_type not in Config.CONST_TYPES[Config.DOMAIN_NAME] and (relation.parameter2_type not in Config.OBJECT_NAME)) or ((relation.parameter2 in param_mapping and relation.parameter2_type not in Config.CONST_TYPES[Config.DOMAIN_NAME]) and (relation.parameter1_type not in Config.OBJECT_NAME))) and (relation.parameter1 != relation.parameter2)) or (((relation.parameter1_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[1]) in robot_id_set) or (relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter2.split("_")[1]) in robot_id_set)) and relation.cr != 0):
                if relation.parameter1 not in param_mapping: 
                    if relation.parameter1_type not in additional_param_objects:
                        additional_param_objects[relation.parameter1_type] = []
                    if relation.parameter1 not in additional_param_objects[relation.parameter1_type]:
                        additional_param_objects[relation.parameter1_type].append(relation.parameter1)

                if relation.parameter2 not in param_mapping:
                    if relation.parameter2_type not in additional_param_objects: 
                        additional_param_objects[relation.parameter2_type] = []
                    if relation.parameter2 not in additional_param_objects[relation.parameter2_type]: 
                        additional_param_objects[relation.parameter2_type].append(relation.parameter2)
                

        # if len(extra_robot_id_set) > 0:
        #     for relation in sorted_true_set:
        #         lr = relation.get_lifted_relation()
        #         if (((relation.parameter1_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[1]) in extra_robot_id_set) or (relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter2.split("_")[1]) in extra_robot_id_set)) and relation.cr != 0) or (relation.parameter1_type in Config.ROBOT_TYPES and relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[-1]) in extra_robot_id_set and int(relation.parameter1.split("_")[-1]) in extra_robot_id_set):
        #             if relation.parameter1 not in param_mapping: 
        #                 if relation.parameter1_type not in additional_param_objects:
        #                     additional_param_objects[relation.parameter1_type] = []
        #                 if relation.parameter1 not in additional_param_objects[relation.parameter1_type]:
        #                     additional_param_objects[relation.parameter1_type].append(relation.parameter1)

        #             if relation.parameter2 not in param_mapping:
        #                 if relation.parameter2_type not in additional_param_objects: 
        #                     additional_param_objects[relation.parameter2_type] = []
        #                 if relation.parameter2 not in additional_param_objects[relation.parameter2_type]: 
        #                     additional_param_objects[relation.parameter2_type].append(relation.parameter2)
        
        param_objects = set(param_mapping.keys())
        param_objects = LiftedPDDLAction.get_param_objects(param_objects,additional_param_objects)
        for relation in cluster[0][0].true_set: 
            lr = relation.get_lifted_relation()
            if relation not in changed_relations:
                # if (((relation.parameter1 in param_mapping and relation.parameter1_type not in Config.CONST_TYPES[Config.DOMAIN_NAME] and (relation.parameter2_type not in Config.OBJECT_NAME)) or ((relation.parameter2 in param_mapping and relation.parameter2_type not in Config.CONST_TYPES[Config.DOMAIN_NAME]) and (relation.parameter1_type not in Config.OBJECT_NAME))) and (relation.parameter1 != relation.parameter2)) or (((relation.parameter1_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[1]) in robot_id_set) or (relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter2.split("_")[1]) in robot_id_set)) and relation.cr != 0): #CANDIDATE
                if set([relation.parameter1,relation.parameter2]).issubset(param_objects):
                    if relation.parameter1 in param_mapping: 
                        pid1 = param_mapping[relation.parameter1]
                    else:
                        if relation.parameter1 not in additional_param_mappings: 
                            pid1 = additional_param_objects[relation.parameter1_type].index(relation.parameter1)+1
                            additional_param_mappings[relation.parameter1] = relation.parameter1_type + "_" +  "extra" + "_p" + str(pid1)
                        pid1 = additional_param_mappings[relation.parameter1]
                        # if len(robot_id_set) > 0:
                        #     if (relation.parameter1_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[1]) in robot_id_set):
                        #         if "extra" in pid1:
                        #             pid1 = pid1.split("_")[0] + "_" + pid1.split("_")[-1]

                    if relation.parameter2 in param_mapping:
                        pid2 = param_mapping[relation.parameter2]
                    else:
                        if relation.parameter2 not in additional_param_mappings: 
                            pid2 = additional_param_objects[relation.parameter2_type].index(relation.parameter2)+1
                            additional_param_mappings[relation.parameter2] = relation.parameter2_type + "_" +  "extra" + "_p" + str(pid2)
                        pid2 = additional_param_mappings[relation.parameter2]
                        # if len(robot_id_set) > 0:
                        #     if (relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter2.split("_")[1]) in robot_id_set):
                        #         if "extra" in pid2:
                        #             pid2 = pid2.split("_")[0] + "_" + pid2.split("_")[-1]

                    parameterized_relation = ParameterizedLiftedRelation(pid1,pid2,lr)
                    common_relations.add(parameterized_relation)

        # if len(extra_robot_id_set) > 0:
        #     for relation in cluster[0][0].true_set:
        #         lr = relation.get_lifted_relation()
        #         # if (((relation.parameter1_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[1]) in extra_robot_id_set) or (relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter2.split("_")[1]) in extra_robot_id_set)) and relation.cr != 0) or (relation.parameter1_type in Config.ROBOT_TYPES and relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[-1]) in extra_robot_id_set and int(relation.parameter1.split("_")[-1]) in extra_robot_id_set): #CANDIDATE
        #         if set([relation.parameter1,relation.parameter2]).issubset(param_objects):
        #             if relation.parameter1 in param_mapping: 
        #                 pid1 = param_mapping[relation.parameter1]
        #             else:
        #                 if relation.parameter1 not in additional_param_mappings:
        #                     pid1 = additional_param_objects[relation.parameter1_type].index(relation.parameter1)+1
        #                     additional_param_mappings[relation.parameter1] = relation.parameter1_type + "_" +  "extra" + "_p" + str(pid1)
        #                 pid1 = additional_param_mappings[relation.parameter1]

        #             if relation.parameter2 in param_mapping:
        #                 pid2 = param_mapping[relation.parameter2]
        #             else:
        #                 if relation.parameter2 not in additional_param_mappings: 
        #                     pid2 = additional_param_objects[relation.parameter2_type].index(relation.parameter2)+1
        #                     additional_param_mappings[relation.parameter2] = relation.parameter2_type + "_" +  "extra" + "_p" + str(pid2)
        #                 pid2 = additional_param_mappings[relation.parameter2]

        #             parameterized_relation = ParameterizedLiftedRelation(pid1,pid2,lr)
        #             common_relations.add(parameterized_relation)

        for transition in cluster[1:]:
            state1, state2 = transition
            local_changed = set()

            for r1 in state1.true_set: 
                if r1 not in state2.true_set:
                    local_changed.add(r1)
            for r1 in state2.true_set: 
                if r1 not in state1.true_set: 
                    local_changed.add(r1)

            local_additional_param_mappings = { }
            additional_param_ids = { }
            relation_set = set()
            local_param_mapping = { }
            robot_id_set = set([])
            extra_robot_id_set = set([])
            local_param_objects = set([])
                
            local_additional_param_objects = {}
            local_sorted_true_set = list(state1.true_set)
            local_sorted_true_set.sort()
            local_sorted_true_set = local_sorted_true_set[::-1]

            local_changed = list(local_changed)
            local_changed.sort()

            lifted_local_changed_set = set() 
            for relation in local_changed: 
                # if relation.cr != 0: 
                lr = relation.get_lifted_relation()
                if len(relation_param_mapping[lr]) == 1:
                    lr_index = 0
                else: 
                    lr_index = -1
                    # print "Ideally it should not even come here..." 
                    if relation.parameter1 in local_param_mapping and relation.parameter2 not in local_param_mapping: 
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if ps[0] == local_param_mapping[relation.parameter1]:
                                lr_index = lr_i
                                break

                    elif relation.parameter1 not in local_param_mapping and relation.parameter2 in local_param_mapping:
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if ps[1] == local_param_mapping[relation.parameter2]:
                                lr_index = lr_i
                                break

                    else:
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if ps[0] == param_mapping[relation.parameter1] and ps[1] == param_mapping[relation.parameter2]: 
                                lr_index = lr_i 
                                break 

                    if lr_index == -1: 
                        print("It should never come here..")
                        print("something is wrong!!")
                        exit (-1) 
                
                pid1 = copy.deepcopy(relation_param_mapping[lr][lr_index][0])
                pid2 = copy.deepcopy(relation_param_mapping[lr][lr_index][1])                                          
                parameterized_relation = ParameterizedLiftedRelation(pid1,pid2,lr)
                                    
                if relation in local_sorted_true_set:
                    relation_set.add(parameterized_relation)
                lifted_local_changed_set.add(parameterized_relation)

                if relation.parameter1 not in local_param_mapping:
                    local_param_mapping[relation.parameter1] = pid1
                if relation.parameter2 not in local_param_mapping:
                    local_param_mapping[relation.parameter2] = pid2      


            for relation in local_sorted_true_set:
                if relation not in local_changed:
                    lr = relation.get_lifted_relation()
                    # if (((relation.parameter1 in local_param_mapping and relation.parameter1_type not in Config.CONST_TYPES[Config.DOMAIN_NAME] and (relation.parameter2_type not in Config.OBJECT_NAME)) or (relation.parameter2 in local_param_mapping and relation.parameter2_type not in Config.CONST_TYPES[Config.DOMAIN_NAME] and (relation.parameter1_type not in Config.OBJECT_NAME))) and relation.parameter1 != relation.parameter2) or (((relation.parameter1_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[1]) in robot_id_set) or (relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter2.split("_")[1]) in robot_id_set)) and relation.cr != 0):
                    if relation.parameter1 not in local_param_mapping:
                        if relation.parameter1_type not in local_additional_param_objects:
                            local_additional_param_objects[relation.parameter1_type] = []
                        if relation.parameter1 not in local_additional_param_objects[relation.parameter1_type]:
                            local_additional_param_objects[relation.parameter1_type].append(relation.parameter1)

                    if relation.parameter2 not in local_param_mapping:
                        if relation.parameter2_type not in local_additional_param_objects:
                            local_additional_param_objects[relation.parameter2_type] = []
                        if relation.parameter2 not in local_additional_param_objects[relation.parameter2_type]:
                            local_additional_param_objects[relation.parameter2_type].append(relation.parameter2)

            # if len(extra_robot_id_set) > 0:
            #     for relation in local_sorted_true_set:
            #         lr = relation.get_lifted_relation()
            #         if (((relation.parameter1_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[1]) in extra_robot_id_set) or (relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter2.split("_")[1]) in extra_robot_id_set)) and relation.cr != 0) or (relation.parameter1_type in Config.ROBOT_TYPES and relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[-1]) in extra_robot_id_set and int(relation.parameter1.split("_")[-1]) in extra_robot_id_set):
            #             if relation.parameter1 not in local_param_mapping: 
            #                 if relation.parameter1_type not in local_additional_param_objects:
            #                     local_additional_param_objects[relation.parameter1_type] = []
            #                 if relation.parameter1 not in local_additional_param_objects[relation.parameter1_type]:
            #                     local_additional_param_objects[relation.parameter1_type].append(relation.parameter1)

            #             if relation.parameter2 not in local_param_mapping:
            #                 if relation.parameter2_type not in local_additional_param_objects: 
            #                     local_additional_param_objects[relation.parameter2_type] = []
            #                 if relation.parameter2 not in local_additional_param_objects[relation.parameter2_type]: 
            #                     local_additional_param_objects[relation.parameter2_type].append(relation.parameter2)

            local_param_objects = set(local_param_mapping.keys())            
            local_param_objects = LiftedPDDLAction.get_param_objects(local_param_objects,local_additional_param_objects)
            for relation in state1.true_set:
                if relation not in local_changed:
                    lr = relation.get_lifted_relation()
                    # if (((relation.parameter1 in local_param_mapping and relation.parameter1_type not in Config.CONST_TYPES[Config.DOMAIN_NAME] and (relation.parameter2_type not in Config.OBJECT_NAME)) or (relation.parameter2 in local_param_mapping and relation.parameter2_type not in Config.CONST_TYPES[Config.DOMAIN_NAME] and (relation.parameter1_type not in Config.OBJECT_NAME))) and relation.parameter1 != relation.parameter2) or (((relation.parameter1_type in Config.ROBOT_TYPES and int(relation.parameter1.split("_")[1]) in robot_id_set) or (relation.parameter2_type in Config.ROBOT_TYPES and int(relation.parameter2.split("_")[1]) in robot_id_set)) and relation.cr != 0): #CANDIDATE
                    if set([relation.parameter1,relation.parameter2]).issubset(local_param_objects):
                        if relation.parameter1 in local_param_mapping:
                            pid1 = local_param_mapping[relation.parameter1]
                        else:
                            if relation.parameter1 not in local_additional_param_mappings:
                                pid1 = local_additional_param_objects[relation.parameter1_type].index(relation.parameter1) + 1
                                local_additional_param_mappings[relation.parameter1] = relation.parameter1_type + "_" +  "extra" + "_p" + str(pid1)
                            pid1 = local_additional_param_mappings[relation.parameter1]
                            
                        if relation.parameter2 in local_param_mapping:
                            pid2 = local_param_mapping[relation.parameter2]
                        else:
                            if relation.parameter2 not in local_additional_param_mappings:
                                pid2 = local_additional_param_objects[relation.parameter2_type].index(relation.parameter2)+1
                                local_additional_param_mappings[relation.parameter2] = relation.parameter2_type + "_" +  "extra" + "_p" + str(pid2)
                            pid2 = local_additional_param_mappings[relation.parameter2]
                            
                        parameterized_relation = ParameterizedLiftedRelation(pid1,pid2,lr)
                        relation_set.add(parameterized_relation)

            new_set = set()
            for relation in relation_set: 
                for relation2 in common_relations: 
                    if relation == relation2:
                        new_set.add(relation)
                        break
            common_relations = copy.deepcopy(new_set)

        common_relations_to_remove = set()
        for re in common_relations:
            extra_flag = (("extra" in re.pid1) or ("extra" in re.pid2))
            if extra_flag:
                pid1 = re.pid1
                # if re.pid1.split("_")[0] in Config.ROBOT_TYPES:
                #     pid1 = re.pid1.split("_")[0] + "_extra_" + re.pid1.split("_")[-1]
                if pid1 in additional_param_mappings.values(): 
                    for o in additional_param_mappings.keys():
                        if additional_param_mappings[o] == pid1:
                            o1 = o 
                            break
                else:
                    for o in param_mapping.keys():
                        if param_mapping[o] == re.pid1:
                            o1 = o 
                            break
                
                pid2 = re.pid2
                # if re.pid2.split("_")[0] in Config.ROBOT_TYPES:
                #     pid2 = re.pid2.split("_")[0] + "_extra_" + re.pid2.split("_")[-1]
                if re.pid2 in additional_param_mappings.values(): 
                    for o in additional_param_mappings.keys():
                        if additional_param_mappings[o] == pid2:
                            o2 = o 
                            break
                else:
                    for o in param_mapping.keys():
                        if param_mapping[o] == re.pid2:
                            o2 = o 
                            break

                # TODO: check this two lines    
                l1 = Link(o1,re.parent_relation.parameter1_type)
                l2 = Link(o2,re.parent_relation.parameter2_type)
                gr = re.parent_relation.get_grounded_relation(l1,l2)

                # if gr not in added_relations[cluster[0][0]]:
                #     common_relations_to_remove.add(re)
                # common_relations_to_remove.add(re)

        for re in common_relations_to_remove:
            common_relations.remove(re)

        lifted_changed_relations = set([a.get_lifted_relation() for a in changed_relations])
        parameterized_changed_relations = set([a for a in common_relations.union(cluster_e_add) if a.parent_relation in lifted_changed_relations])

        # aux_to_add = set()
        # for re in common_relations:
        #     for ax in all_aux:
        #         if re.parameter1_type == ax.parameter1_type and re.parameter2_type == ax.parameter2_type and re.cr != ax.cr:
        #             aux_to_add.add(re)
    
        param_set = set()
        for relation in common_relations: 
            # if relation.parent_relation.parameter1_type not in Config.CONST_TYPES[Config.DOMAIN_NAME]:
            param1 = Parameter(relation.pid1,relation.parent_relation.parameter1_type)
            param_set.add(param1)

            # if relation.parent_relation.parameter2_type not in Config.CONST_TYPES[Config.DOMAIN_NAME]:
            param2 = Parameter(relation.pid2,relation.parent_relation.parameter2_type)
            param_set.add(param2)

        
        for relation in cluster_e_add:
            # if relation.parent_relation.parameter1_type not in Config.CONST_TYPES[Config.DOMAIN_NAME]:
            param1 = Parameter(relation.pid1,relation.parent_relation.parameter1_type)
            param_set.add(param1)

            # if relation.parent_relation.parameter2_type not in Config.CONST_TYPES[Config.DOMAIN_NAME]:
            param2 = Parameter(relation.pid2,relation.parent_relation.parameter2_type)
            param_set.add(param2)

        for relation in cluster_e_delete:
            # if relation.parent_relation.parameter1_type not in Config.CONST_TYPES[Config.DOMAIN_NAME]:
            param1 = Parameter(relation.pid1,relation.parent_relation.parameter1_type)
            param_set.add(param1)

            # if relation.parent_relation.parameter2_type not in Config.CONST_TYPES[Config.DOMAIN_NAME]:
            param2 = Parameter(relation.pid2,relation.parent_relation.parameter2_type)
            param_set.add(param2)

        preconditions = LiftedPDDLPrecondition(true_set=common_relations, false_set=set(),true_aux_set=set())
        effects = LiftedPDDLEffect(cluster_e_add,cluster_e_delete,set(), set())
        LiftedPDDLAction.action_id += 1

        return LiftedPDDLAction(LiftedPDDLAction.action_id, list(param_set), preconditions, effects)
    
    def __str__(self):
        s = "(:action a{} \n".format(self.action_id)
        param_string = ""
        for param in self.parameters:
            param_string += " ?{} - {} ".format(param.pid,param.type)
        s += ":parameters ({})\n".format(param_string)
        precondition_string = ""
        for i,param in enumerate(self.parameters):
            for j,param2 in enumerate(self.parameters):
                if j>i and param.type == param2.type:
                    precondition_string += "\t(not (= ?{} ?{}))\n".format(param.pid,param2.pid)
        
        precondition_string+=str(self.preconditions)

        required_parameter_str = ""
        # for re in self.effects.add_set:
        #     if "goalLoc" in str(re.pid1) and re.parent_relation.cr == 1:
        #         required_parameter_str = str(re.pid2)
        #         break

        required_planks_str = ""
        for p1,p2 in self.required_planks:
            required_planks_str += "\t(or (goalLoc_1 goalLoc_Const {}) (not (= ?{}  {} )))\n".format(p1,required_parameter_str,p2)

        precondition_string+=required_planks_str

        states_to_neglect_str = ""
        for state in self.states_to_neglect:
            state_string = ""
            for i,prop in enumerate(state.true_set): 
                state_string += str(prop)
                state_string += " "
                if i > 0 and i % 4 == 0: 
                    state_string += "\n"
            states_to_neglect_str += "(not (and {}))\n".format(state_string)
        
        precondition_string += states_to_neglect_str

        s += ":precondition (and \n{}) \n".format(precondition_string)

        effect_string = str(self.effects)

        s += ":effect (and \n {} ) \n".format(effect_string)
        s+= ")\n"

        return s
    
    def get_grounded_action(self, grounding,lifted_action_id): 
        grounded_precondition = self.preconditions.get_grounded_precondition(grounding)
        grounded_effect = self.effects.get_grounded_effect(grounding)
        return GroundedPDDLAction(grounded_precondition,grounded_effect,lifted_action_id)

    def __eq__(self,o):
        if (self.preconditions != o.preconditions) or (self.effects != o.effects): #or (self.parameters != o.parameters):
            return False
        return True
    
    def __lt__(self,o):
        return self.__str__() < o.__str__()
    
    def __hash__(self):
        return hash(self.__str__().split(":parameters")[-1])

class GroundedPDDLAction(object):
    def __init__(self,precondition,effect,lifted_action_id):
        self.precondition = precondition
        self.effect = effect 
        self.changed_relations = self.get_changed_relations()
        self.lifted_action_id = lifted_action_id
        self.sampling_region = None 

    def check_applicability(self,pddl_state):
        return self.precondition.check_in_state(pddl_state)

    def apply(self,pddl_state):
        return self.effect.apply(pddl_state)

    def get_changed_relations(self):
        changed_relations = []
        for rel in list(self.precondition.true_set):
            if rel in self.effect.delete_set:
                changed_relations.append(rel)

        for rel in list(self.precondition.false_set):
            if rel in self.effect.add_set:
                changed_relations.append(rel)

        return changed_relations
        
    # def compute_refinement(self,env_state,sim_object,previous_instances,action_info,compute_motion_plan=True): 
        traj = None
        flag = 0
        current_grab_flag = 0
        grab_relation = None
        sim_object.set_env_state(env_state)

        grabbed = False
        for r in range(1,env_state.num_robots+1):
            grabbed = (grabbed or getattr(env_state,"grabbed_flag_{}".format(r)))
                                            
        if grabbed:
            for re in self.changed_relations:
                grabbed_object_flag = False
                for r in range(1,env_state.num_robots+1):
                    if (re.parameter1 == getattr(env_state,"grabbed_object_{}".format(r)) or re.parameter2 == getattr(env_state,"grabbed_object_{}".format(r))):
                        grabbed_object_flag = True
                if (Config.GRIPPER_NAME in re.parameter1 or Config.GRIPPER_NAME in re.parameter2) and grabbed_object_flag:
                    grab_relation = re
                    if (re.parameter1_type == Config.GRIPPER_NAME):
                        id = re.parameter1.split("_")[1]
                        # grabbed = False
                        # for n in range(1,env_state.num_robots+1):
                        #     grabbed = (grabbed or getattr(env_state,"grabbed_flag_{}".format(n)))
                        
                        if grabbed:
                            if re.parameter2 == getattr(env_state,"grabbed_object_{}".format(id)):
                                current_grab_flag = 1
                            else:
                                current_grab_flag = 2
                        else:
                            current_grab_flag = 0

                    elif (re.parameter2_type == Config.GRIPPER_NAME):
                        id = re.parameter2.split("_")[1]
                        # grabbed = False
                        # for n in range(1,env_state.num_robots+1):
                        #     grabbed = (grabbed or getattr(env_state,"grabbed_flag_{}".format(n)))
                        
                        if grabbed:
                            if re.parameter1 == getattr(env_state,"grabbed_object_{}".format(id)):
                                current_grab_flag = 1
                            else:
                                current_grab_flag = 2
                        else:
                            current_grab_flag = 0                  

                    break
        grab_bool = None 
        used_relations = set([])
        relation_selected = False
        objects_in_collision_list = []
        objects_in_collision_flag = False
        while not (relation_selected or objects_in_collision_flag):
            relation_options = self.effect.add_set.difference(used_relations)
            if len(relation_options) == 0:
                break

            relation_to_use = None
            robot_re_flag = False
            grabbed = False
            for r in range(1,env_state.num_robots+1):
                grabbed = (grabbed or getattr(env_state,"grabbed_flag_{}".format(r)))
            
            for re in relation_options:
                if re.parameter1.split("_")[0] in Config.IMMOVABLE_OBJECTS or re.parameter2.split("_")[0] in Config.IMMOVABLE_OBJECTS: 
                    relation_to_use = re
                    break
                
                if grabbed:
                    grabbed_object_flag = False
                    for r in range(1,env_state.num_robots+1):
                        if (re.parameter1 == getattr(env_state,"grabbed_object_{}".format(r)) or re.parameter2 == getattr(env_state,"grabbed_object_{}".format(r))):
                            grabbed_object_flag = True

                    if (Config.GRIPPER_NAME in re.parameter1 or Config.GRIPPER_NAME in re.parameter2): #and grabbed_object_flag:
                        relation_to_use = re
                        break
                    if grabbed_object_flag:
                        relation_to_use = re

                else:
                    if re.parameter1.split("_")[0] in Config.ROBOT_TYPES.keys() or re.parameter2.split("_")[0] in Config.ROBOT_TYPES.keys():
                        relation_to_use = re
                        robot_re_flag = True
                        break

            if relation_to_use is None:
                relation_to_use = np.random.choice(list(relation_options))

            valid_sample_found = False
            try_new_region = False

            while True: 
                sim_object.set_env_state(env_state)
                if self.sampling_region is None or try_new_region: 
                    try: 
                        self.sampling_region = relation_to_use.sample_region()
                    except StopIteration: 
                        # relation out of regions. move to different relation
                        valid_sample_found = False
                        try_new_region = False
                        break 
                    else: 
                        relation_to_use.init_sample_generator(env_state, sim_object, self.sampling_region,action_info)
                while True: 
                    objects_in_collision_flag = False
                    try: 
                        next_sample,lifted_region_used,object_with_transform,sampled_refined_grounded_pose,rob, static_list, eef_transform,objects_in_collision_list, delta_mp = relation_to_use.get_next_sample()
                    except StopIteration: 
                        # region out of samples. move to different region 
                        valid_sample_found = False
                        try_new_region = True
                        if len(objects_in_collision_list) > 0:
                            objects_in_collision_flag = True
                        break
                    
                    grab_bool = None 
                    static_object = static_list[0]
                    if len(next_sample) == 1:
                        if len(objects_in_collision_list) == 0:
                            continue
                        else:
                            objects_in_collision_flag = True
                            valid_sample_found = False
                            break

                    #jayesh's some book keeping
                    if static_list[0].split("_")[0] in Config.CONST_TYPES[Config.DOMAIN_NAME]:
                        static_object,static_num = static_list
                        param_index = 2-static_num
                        param_list = [relation_to_use.parameter1,relation_to_use.parameter2]
                        other_param = param_list[param_index]
                        static_object = static_object.split("_")[0] + "_" +other_param.split("_")[1]
                    
                    if grab_relation is None:
                        next_grab_flag = next_sample[-1]
                    else:
                        if len(relation_to_use.region)==1:
                            sampled_lifted_region = relation_to_use.region[0]
                        else:
                            index = np.random.randint(len(relation_to_use.region))
                            sampled_lifted_region = relation_to_use.region[index]
                        next_grab_flag = sampled_lifted_region[-1]         
                    if (not (bool(current_grab_flag and next_grab_flag))) and (2 not in [current_grab_flag,next_grab_flag]):
                        grab_action = current_grab_flag - next_grab_flag

                        if grab_action >0:
                            grab_bool=False
                        elif grab_action <0:
                            grab_bool=True
                        else:
                            grab_bool = None                  
                    else:
                        grab_bool = None

                    # check if this sample is valid or not 

                    satisfactory_flag,new_env_state,failed_relation = self.satisfactory_sample(object_with_transform,sampled_refined_grounded_pose,env_state,grab_bool,sim_object,rob,next_sample,delta_mp)

                    if satisfactory_flag: 
                        valid_sample_found = True 
                        break 
                
                if valid_sample_found: 
                    break 
                else:
                    if objects_in_collision_flag:
                        break
            
            if valid_sample_found: 
                if self.motion_plan_flag_check(env_state,relation_to_use,current_grab_flag):
                    counter = 0
                    while traj is None:
                        if compute_motion_plan:
                            index = len(next_sample) - 1 
                            traj = sim_object.compute_motion_plan(goal=next_sample[:index],robot=rob.robot)
                        else:
                            traj = next_sample

                        counter+=1
                        if counter >= Config.MP_MAX_COUNT and traj is None:
                            flag = 1
                            # relation_selected = True
                            break
                # found valid sample.. try motion planning 

            else: 
                if not objects_in_collision_flag:
                    used_relations.add(relation_to_use)

            if traj is not None or grab_bool is not None:
                relation_selected = True
                break
            
        if len(relation_options) == 0:
            return None, None, (None, None, None), None, None, None, None, []
        
        lifted_region_used_sampling_count = 1
        if relation_to_use in previous_instances.keys():
            for (prev_region,region_sampling_count) in previous_instances[relation_to_use]:
                if prev_region == tuple(lifted_region_used):
                    region_sampling_count+=1
                    lifted_region_used_sampling_count = region_sampling_count
                    break
                    
        prev_instance_tuple = (relation_to_use,lifted_region_used,lifted_region_used_sampling_count)
        
        if (traj is not None or flag == 1) and grab_bool is not None:
            return None, None, prev_instance_tuple, rob, sampled_refined_grounded_pose, static_object, eef_transform, objects_in_collision_list
        
        if list(action_info) != [None,None,None]:
            if traj is not None:
                traj_list = [traj,delta_mp]
                grab_list = None
            if grab_bool is not None:
                grab_list = [grab_bool,delta_mp]
                traj_list = None
        
        else:
            traj_list = traj
            grab_list = grab_bool

        return traj_list,grab_list,prev_instance_tuple, rob, sampled_refined_grounded_pose, static_object, eef_transform, objects_in_collision_list
    
    # def satisfactory_sample(self,object_with_transform,sampled_refined_grounded_pose,env_state,grab_bool,sim_object,rob,robot_dof_vals,delta_mp=None):        
        grabbed_obj_name = None
        if grab_bool is not None:
            if grab_bool:
                for relation in self.effect.add_set:
                    if Config.GRIPPER_NAME in relation.parameter1:
                        if relation.parameter2.split("_")[0] not in Config.IMMOVABLE_OBJECTS and relation.parameter2.split("_")[0] != Config.GRIPPER_NAME:
                            grabbed_obj_name = relation.parameter2
                            id = relation.parameter1.split("_")[1]
            else:
                grabbed_obj_name = getattr(env_state,"grabbed_object_{}".format(rob.id))
            
            traj = grab_bool
        else:
            traj = robot_dof_vals[:-1]
        
        new_env_state = sim_object.execute_refinement(traj=traj,robot=rob,obj_name=grabbed_obj_name)
        if delta_mp is not None and type(traj) != bool:
            new_env_state = sim_object.execute_refinement(traj=delta_mp,robot=rob,obj_name=grabbed_obj_name)

        # if (self.check_collisions_in_env_state(sim_object=sim_object,env_state=new_env_state)):
        #     return False, new_env_state, None

        for relation in self.effect.add_set:
            if relation.evaluate_in_ll_state(new_env_state) is not True:
                sim_object.set_env_state(env_state)
                return False,new_env_state,relation
            # else:
            #     print("debug")
        
        sim_object.set_env_state(env_state)
        return True,new_env_state,None

    # def motion_plan_flag_check(self,env_state,relation_to_use,current_grab_flag):
        new_region_list = []
        new_relation = copy.deepcopy(relation_to_use)
        for re in new_relation.region:
            new_region = re[:-1]
            new_region.append(int(current_grab_flag))
            new_region_list.append(new_region)

        new_relation.region = new_region_list

        return not (new_relation.evaluate_in_ll_state(env_state))

    # def check_collisions_in_env_state(self,sim_object,env_state):
        current_env_state = sim_object.get_current_state()
        sim_object.set_env_state(env_state)
        collision_flag = sim_object.collision_check(env_state.object_dict.keys())

        sim_object.set_env_state(current_env_state)
        return collision_flag
        
    def __str__(self):
        add_string = "adding -> "
        if len(self.effect.add_set) > 0:
            for rel in self.effect.add_set:
                add_string += rel.__str__()
                add_string += ", "
        else:
            add_string += "NOTHING"
        
        delete_string = "|| deleting -> "
        if len(self.effect.delete_set) > 0:
            for rel in self.effect.delete_set:
                delete_string += rel.__str__()
                delete_string += ", "
        else:
            delete_string += "NOTHING"
        
        id_string = "lifted_action_id:{};".format(self.lifted_action_id)
        return id_string+add_string+delete_string
    
if __name__ == "__main__":
    # test data structures
    # start with the base one

    # lifted relation
    at_relation = Relation("object", "location", "At")
    close_to_relation = Relation("robot", "location", "CloseTo")
    # ground relation
    obj2pid = {
        "Robot": 0,
        "Apple": 1,
        "Table": 2,
        "Banana": 3
    }

    robot_param = Parameter(obj2pid["Robot"], "robot", "Robot")
    apple_param = Parameter(obj2pid["Apple"], "object", "Apple")
    table_param = Parameter(obj2pid["Table"], "location", "Table")
    banana_param = Parameter(obj2pid["Banana"], "object", "Banana")

    at_relation_grounded_apple_table = at_relation.get_grounded_relation(apple_param, table_param)
    at_relation_grounded_banana_table = at_relation.get_grounded_relation(banana_param, table_param)

    close_to_relation_grounded_robot_table = close_to_relation.get_grounded_relation(robot_param, table_param)
    # transition 0
    # PDDL state 0
    true_set = {close_to_relation_grounded_robot_table, at_relation_grounded_banana_table}
    false_set  = {at_relation_grounded_apple_table}
    grounded_state_0 = PDDLState(true_set, false_set)
    # PDDL state 1
    true_set = {close_to_relation_grounded_robot_table, at_relation_grounded_apple_table, at_relation_grounded_banana_table}
    false_set  = {}
    grounded_state_1 = PDDLState(true_set, false_set)
    transition_0 = [grounded_state_0, grounded_state_1]

    # transition 1
    # PDDL state 0
    true_set = {close_to_relation_grounded_robot_table}
    false_set  = {at_relation_grounded_apple_table,  at_relation_grounded_banana_table}
    grounded_state_0 = PDDLState(true_set, false_set)
    # PDDL state 1
    true_set = {close_to_relation_grounded_robot_table, at_relation_grounded_apple_table}
    false_set  = {at_relation_grounded_banana_table}
    grounded_state_1 = PDDLState(true_set, false_set)
    transition_1 = [grounded_state_0, grounded_state_1]

    # test cluster: list[list[PDDLState, PDDLState]]
    cluster = [
        transition_0,
        transition_1
    ]
    operator = LiftedPDDLAction.get_action_from_cluster(cluster)
    breakpoint()