from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from itertools import product

from data_structure import Predicate, PredicateState, Skill

from skillwrapper.refactored.operators import Effects, Preconditions
from skillwrapper.refactored.parameters import DiscreteParameter

# TODO: Replace PDDLState with AbstractState
# TODO: Replace Relation with Predicate, I think
# TODO: Replace GroundedRelation with PredicateInstance, I think
# TODO: Remove mentions of Parameter


@functools.total_ordering
class LiftedPDDLAction:
    action_id = 0

    def __init__(
        self,
        id,
        parameters,
        preconditions,
        effects,
    ):
        self.action_id = id
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects

    @staticmethod
    def get_param_objects(param_objects_set, additional_param_objects_dict):
        param_objects = copy.deepcopy(param_objects_set)
        for obj_type in additional_param_objects_dict.keys():
            param_objects = param_objects.union(set(additional_param_objects_dict[obj_type]))

        return param_objects

    @staticmethod
    def get_action_from_cluster(cluster, param_ids={}):
        # cluster: list[list[AbstractState, AbstractState]]

        cluster_e_add = set()
        cluster_e_delete = set()
        changed_relations = set()

        temp_added = set()
        temp_deleted = set()

        for r1 in cluster[0][0].true_set:  # Index: trans, (pre/post)
            if r1 not in cluster[0][1].true_set:
                changed_relations.add(r1)
                temp_deleted.add(r1)
        for r1 in cluster[0][1].true_set:
            if r1 not in cluster[0][0].true_set:
                changed_relations.add(r1)
                temp_added.add(r1)

        param_mapping = {}
        relation_param_mapping = {}

        for relation in changed_relations:
            if relation.parameter1_type not in param_ids:
                param_ids[relation.parameter1_type] = 1
            if relation.parameter2_type not in param_ids:
                param_ids[relation.parameter2_type] = 1
            if relation.parameter1 not in param_mapping:
                if relation.parameter1 in param_ids:
                    pid1 = param_ids[relation.parameter1]
                else:
                    pid1 = param_ids[relation.parameter1_type]
                    param_ids[relation.parameter1_type] += 1
                param_mapping[relation.parameter1] = relation.parameter1_type + "_p" + str(pid1)
            if relation.parameter2 not in param_mapping:
                if relation.parameter2 in param_ids:
                    pid2 = param_ids[relation.parameter2]
                else:
                    pid2 = param_ids[relation.parameter2_type]
                    param_ids[relation.parameter2_type] += 1
                param_mapping[relation.parameter2] = relation.parameter2_type + "_p" + str(pid2)
            lr = relation.get_lifted_relation()
            if lr not in relation_param_mapping:
                relation_param_mapping[lr] = [
                    [param_mapping[relation.parameter1], param_mapping[relation.parameter2]],
                ]
            else:
                relation_param_mapping[lr].append(
                    [param_mapping[relation.parameter1], param_mapping[relation.parameter2]],
                )

        # NOTE: I don't know what do these temp_* means. I commented them out and the operator looks good now
        for relation in temp_added:
            lifted_relation = relation.get_lifted_relation()
            pid1 = param_mapping[relation.parameter1]
            pid2 = param_mapping[relation.parameter2]
            cluster_e_add.add(LiftedRelation((pid1, pid2), lifted_relation))  # <-

        for relation in temp_deleted:
            lifted_relation = relation.get_lifted_relation()
            pid1 = param_mapping[relation.parameter1]
            pid2 = param_mapping[relation.parameter2]
            cluster_e_delete.add(LiftedRelation((pid1, pid2), lifted_relation))  # <-

        relations_union = cluster[0][0].true_set.union(cluster[0][0].false_set)
        for relation in temp_added:
            p1 = relation.parameter1
            p2 = relation.parameter2
            for p in relations_union:
                if p.parameter1 == p1 and p.parameter2 == p2 and relation != p:
                    pa = param_mapping[p.parameter1]
                    pb = param_mapping[p.parameter2]
                    lifted_relation = p.get_lifted_relation()
                    parameterized_relation = LiftedRelation((pa, pb), lifted_relation)

        common_relations = set()
        additional_param_mappings = {}
        param_objects = set()

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
                        if (
                            ps[0] == param_mapping[relation.parameter1]
                            and ps[1] == param_mapping[relation.parameter2]
                        ):
                            lr_index = lr_i
                            break
                    if lr_index == -1:
                        print("It should never come here..")
                        print("something is wrong!!")
                        exit(-1)
                pid1 = copy.deepcopy(relation_param_mapping[lr][lr_index][0])
                pid2 = copy.deepcopy(relation_param_mapping[lr][lr_index][1])
                parameterized_relation = LiftedRelation((pid1, pid2), lr)

                common_relations.add(parameterized_relation)

        for relation in sorted_true_set:
            lr = relation.get_lifted_relation()
            if relation not in changed_relations:
                if relation.parameter1 not in param_mapping:
                    if relation.parameter1_type not in additional_param_objects:
                        additional_param_objects[relation.parameter1_type] = []
                    if (
                        relation.parameter1
                        not in additional_param_objects[relation.parameter1_type]
                    ):
                        additional_param_objects[relation.parameter1_type].append(
                            relation.parameter1,
                        )

                if relation.parameter2 not in param_mapping:
                    if relation.parameter2_type not in additional_param_objects:
                        additional_param_objects[relation.parameter2_type] = []
                    if (
                        relation.parameter2
                        not in additional_param_objects[relation.parameter2_type]
                    ):
                        additional_param_objects[relation.parameter2_type].append(
                            relation.parameter2,
                        )

        param_objects = set(param_mapping.keys())
        param_objects = LiftedPDDLAction.get_param_objects(param_objects, additional_param_objects)
        for relation in cluster[0][0].true_set:
            lr = relation.get_lifted_relation()
            if relation not in changed_relations:
                if set([relation.parameter1, relation.parameter2]).issubset(param_objects):
                    if relation.parameter1 in param_mapping:
                        pid1 = param_mapping[relation.parameter1]
                    else:
                        if relation.parameter1 not in additional_param_mappings:
                            if relation.parameter1 in param_ids:
                                pid1 = param_ids[relation.parameter1]
                            else:
                                pid1 = (
                                    additional_param_objects[relation.parameter1_type].index(
                                        relation.parameter1,
                                    )
                                    + 1
                                )
                            # additional_param_mappings[relation.parameter1] = relation.parameter1_type + "_" +  "extra" + "_p" + str(pid1)
                            additional_param_mappings[relation.parameter1] = (
                                relation.parameter1_type + "_p" + str(pid1)
                            )
                        pid1 = additional_param_mappings[relation.parameter1]

                    if relation.parameter2 in param_mapping:
                        pid2 = param_mapping[relation.parameter2]
                    else:
                        if relation.parameter2 not in additional_param_mappings:
                            if relation.parameter2 in param_ids:
                                pid2 = param_ids[relation.parameter2]
                            else:
                                pid2 = (
                                    additional_param_objects[relation.parameter2_type].index(
                                        relation.parameter2,
                                    )
                                    + 1
                                )
                            # additional_param_mappings[relation.parameter2] = relation.parameter2_type + "_" +  "extra" + "_p" + str(pid2)
                            additional_param_mappings[relation.parameter2] = (
                                relation.parameter2_type + "_p" + str(pid2)
                            )
                        pid2 = additional_param_mappings[relation.parameter2]

                    parameterized_relation = LiftedRelation((pid1, pid2), lr)
                    common_relations.add(parameterized_relation)

        for transition in cluster[1:]:
            state1, state2 = transition
            local_changed = set()

            for r1 in state1.true_set:
                if r1 not in state2.true_set:
                    local_changed.add(r1)
            for r1 in state2.true_set:
                if r1 not in state1.true_set:
                    local_changed.add(r1)

            local_additional_param_mappings = (
                {} if not param_ids else additional_param_mappings | param_mapping
            )
            relation_set = set()
            local_param_mapping = {} if not param_ids else additional_param_mappings | param_mapping
            local_param_objects = set([])

            local_additional_param_objects = {}
            local_sorted_true_set = list(state1.true_set)
            local_sorted_true_set.sort()
            local_sorted_true_set = local_sorted_true_set[::-1]

            local_changed = list(local_changed)
            local_changed.sort()

            lifted_local_changed_set = set()
            for relation in local_changed:
                lr: Relation = relation.get_lifted_relation()
                if len(relation_param_mapping[lr]) == 1:
                    lr_index = 0
                else:
                    lr_index = -1
                    # print "Ideally it should not even come here..."
                    if (
                        relation.parameter1 in local_param_mapping
                        and relation.parameter2 not in local_param_mapping
                    ):
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if ps[0] == local_param_mapping[relation.parameter1]:
                                lr_index = lr_i
                                break

                    elif (
                        relation.parameter1 not in local_param_mapping
                        and relation.parameter2 in local_param_mapping
                    ):
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if ps[1] == local_param_mapping[relation.parameter2]:
                                lr_index = lr_i
                                break

                    else:
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if (
                                ps[0] == param_mapping[relation.parameter1]
                                and ps[1] == param_mapping[relation.parameter2]
                            ):
                                lr_index = lr_i
                                break

                    if lr_index == -1:
                        print("It should never come here..")
                        print("something is wrong!!")
                        exit(-1)

                pid1 = copy.deepcopy(relation_param_mapping[lr][lr_index][0])
                pid2 = copy.deepcopy(relation_param_mapping[lr][lr_index][1])
                parameterized_relation = LiftedRelation((pid1, pid2), lr)

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
                    if relation.parameter1 not in local_param_mapping:
                        if relation.parameter1_type not in local_additional_param_objects:
                            local_additional_param_objects[relation.parameter1_type] = []
                        if (
                            relation.parameter1
                            not in local_additional_param_objects[relation.parameter1_type]
                        ):
                            local_additional_param_objects[relation.parameter1_type].append(
                                relation.parameter1,
                            )

                    if relation.parameter2 not in local_param_mapping:
                        if relation.parameter2_type not in local_additional_param_objects:
                            local_additional_param_objects[relation.parameter2_type] = []
                        if (
                            relation.parameter2
                            not in local_additional_param_objects[relation.parameter2_type]
                        ):
                            local_additional_param_objects[relation.parameter2_type].append(
                                relation.parameter2,
                            )

            local_param_objects = set(local_param_mapping.keys())
            local_param_objects = LiftedPDDLAction.get_param_objects(
                local_param_objects,
                local_additional_param_objects,
            )
            for relation in state1.true_set:
                if relation not in local_changed:
                    lr = relation.get_lifted_relation()
                    if set([relation.parameter1, relation.parameter2]).issubset(
                        local_param_objects,
                    ):
                        if relation.parameter1 in local_param_mapping:
                            pid1 = local_param_mapping[relation.parameter1]
                        else:
                            if relation.parameter1 not in local_additional_param_mappings:
                                if relation.parameter1 in param_ids:
                                    pid1 = param_ids[relation.parameter1]
                                else:
                                    pid1 = (
                                        local_additional_param_objects[
                                            relation.parameter1_type
                                        ].index(relation.parameter1)
                                        + 1
                                    )
                                # local_additional_param_mappings[relation.parameter1] = relation.parameter1_type + "_" +  "extra" + "_p" + str(pid1)
                                local_additional_param_mappings[relation.parameter1] = (
                                    relation.parameter1_type + "_p" + str(pid1)
                                )
                            pid1 = local_additional_param_mappings[relation.parameter1]

                        if relation.parameter2 in local_param_mapping:
                            pid2 = local_param_mapping[relation.parameter2]
                        else:
                            if relation.parameter2 not in local_additional_param_mappings:
                                if relation.parameter2 in param_ids:
                                    pid2 = param_ids[relation.parameter2]
                                else:
                                    pid2 = (
                                        local_additional_param_objects[
                                            relation.parameter2_type
                                        ].index(relation.parameter2)
                                        + 1
                                    )
                                # local_additional_param_mappings[relation.parameter2] = relation.parameter2_type + "_" +  "extra" + "_p" + str(pid2)
                                local_additional_param_mappings[relation.parameter2] = (
                                    relation.parameter2_type + "_p" + str(pid2)
                                )
                            pid2 = local_additional_param_mappings[relation.parameter2]

                        parameterized_relation = LiftedRelation((pid1, pid2), lr)
                        relation_set.add(parameterized_relation)

            new_set = set()
            for relation in relation_set:
                for relation2 in common_relations:
                    if relation == relation2:
                        new_set.add(relation)
                        break
            common_relations = copy.deepcopy(new_set)

        ########## NEGATIVE PRECONDITION

        neg_common_relations = set()
        additional_param_mappings = {}
        param_objects = set([])

        additional_param_objects = {}
        sorted_false_set = list(cluster[0][0].false_set)
        sorted_false_set.sort()
        sorted_false_set = sorted_false_set[::-1]

        for relation in sorted_false_set:
            lr = relation.get_lifted_relation()
            if relation in changed_relations:
                if len(relation_param_mapping[lr]) == 1:
                    lr_index = 0
                else:
                    lr_index = -1
                    for lr_i in range(len(relation_param_mapping[lr])):
                        ps = relation_param_mapping[lr][lr_i]
                        if (
                            ps[0] == param_mapping[relation.parameter1]
                            and ps[1] == param_mapping[relation.parameter2]
                        ):
                            lr_index = lr_i
                            break
                    if lr_index == -1:
                        print("It should never come here..")
                        print("something is wrong!!")
                        exit(-1)
                pid1 = copy.deepcopy(relation_param_mapping[lr][lr_index][0])
                pid2 = copy.deepcopy(relation_param_mapping[lr][lr_index][1])
                parameterized_relation = LiftedRelation((pid1, pid2), lr)
                neg_common_relations.add(parameterized_relation)

        for relation in sorted_false_set:
            lr = relation.get_lifted_relation()
            if relation not in changed_relations:
                if relation.parameter1 not in param_mapping:
                    if relation.parameter1_type not in additional_param_objects:
                        additional_param_objects[relation.parameter1_type] = []
                    if (
                        relation.parameter1
                        not in additional_param_objects[relation.parameter1_type]
                    ):
                        additional_param_objects[relation.parameter1_type].append(
                            relation.parameter1,
                        )

                if relation.parameter2 not in param_mapping:
                    if relation.parameter2_type not in additional_param_objects:
                        additional_param_objects[relation.parameter2_type] = []
                    if (
                        relation.parameter2
                        not in additional_param_objects[relation.parameter2_type]
                    ):
                        additional_param_objects[relation.parameter2_type].append(
                            relation.parameter2,
                        )

        param_objects = set(param_mapping.keys())
        param_objects = LiftedPDDLAction.get_param_objects(param_objects, additional_param_objects)
        for relation in cluster[0][0].false_set:
            lr = relation.get_lifted_relation()
            if relation not in changed_relations:
                if set([relation.parameter1, relation.parameter2]).issubset(param_objects):
                    if relation.parameter1 in param_mapping:
                        pid1 = param_mapping[relation.parameter1]
                    else:
                        if relation.parameter1 not in additional_param_mappings:
                            if relation.parameter1 in param_ids:
                                pid1 = param_ids[relation.parameter1]
                            else:
                                pid1 = (
                                    additional_param_objects[relation.parameter1_type].index(
                                        relation.parameter1,
                                    )
                                    + 1
                                )
                            # additional_param_mappings[relation.parameter1] = relation.parameter1_type + "_" +  "extra" + "_p" + str(pid1)
                            additional_param_mappings[relation.parameter1] = (
                                relation.parameter1_type + "_p" + str(pid1)
                            )
                        pid1 = additional_param_mappings[relation.parameter1]

                    if relation.parameter2 in param_mapping:
                        pid2 = param_mapping[relation.parameter2]
                    else:
                        if relation.parameter2 not in additional_param_mappings:
                            if relation.parameter2 in param_ids:
                                pid2 = param_ids[relation.parameter2]
                            else:
                                pid2 = (
                                    additional_param_objects[relation.parameter2_type].index(
                                        relation.parameter2,
                                    )
                                    + 1
                                )
                            # additional_param_mappings[relation.parameter2] = relation.parameter2_type + "_" +  "extra" + "_p" + str(pid2)
                            additional_param_mappings[relation.parameter2] = (
                                relation.parameter2_type + "_p" + str(pid2)
                            )
                        pid2 = additional_param_mappings[relation.parameter2]

                    parameterized_relation = LiftedRelation((pid1, pid2), lr)
                    neg_common_relations.add(parameterized_relation)

        for transition in cluster[1:]:
            state1, state2 = transition
            local_changed = set()

            for r1 in state1.false_set:
                if r1 not in state2.false_set:
                    local_changed.add(r1)
            for r1 in state2.false_set:
                if r1 not in state1.false_set:
                    local_changed.add(r1)

            local_additional_param_mappings = (
                {} if not param_ids else additional_param_mappings | param_mapping
            )
            relation_set = set()
            local_param_mapping = {} if not param_ids else additional_param_mappings | param_mapping
            local_param_objects = set([])

            local_additional_param_objects = {}
            local_sorted_false_set = list(state1.false_set)
            local_sorted_false_set.sort()
            local_sorted_false_set = local_sorted_false_set[::-1]

            local_changed = list(local_changed)
            local_changed.sort()

            lifted_local_changed_set = set()
            for relation in local_changed:
                lr = relation.get_lifted_relation()
                if len(relation_param_mapping[lr]) == 1:
                    lr_index = 0
                else:
                    lr_index = -1
                    # print "Ideally it should not even come here..."
                    if (
                        relation.parameter1 in local_param_mapping
                        and relation.parameter2 not in local_param_mapping
                    ):
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if ps[0] == local_param_mapping[relation.parameter1]:
                                lr_index = lr_i
                                break

                    elif (
                        relation.parameter1 not in local_param_mapping
                        and relation.parameter2 in local_param_mapping
                    ):
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if ps[1] == local_param_mapping[relation.parameter2]:
                                lr_index = lr_i
                                break

                    else:
                        for lr_i in range(len(relation_param_mapping[lr])):
                            ps = relation_param_mapping[lr][lr_i]
                            if (
                                ps[0] == param_mapping[relation.parameter1]
                                and ps[1] == param_mapping[relation.parameter2]
                            ):
                                lr_index = lr_i
                                break

                    if lr_index == -1:
                        print("It should never come here..")
                        print("something is wrong!!")
                        exit(-1)

                pid1 = copy.deepcopy(relation_param_mapping[lr][lr_index][0])
                pid2 = copy.deepcopy(relation_param_mapping[lr][lr_index][1])
                parameterized_relation = LiftedRelation((pid1, pid2), lr)

                if relation in local_sorted_false_set:
                    relation_set.add(parameterized_relation)
                lifted_local_changed_set.add(parameterized_relation)

                if relation.parameter1 not in local_param_mapping:
                    local_param_mapping[relation.parameter1] = pid1
                if relation.parameter2 not in local_param_mapping:
                    local_param_mapping[relation.parameter2] = pid2

            for relation in local_sorted_false_set:
                if relation not in local_changed:
                    lr = relation.get_lifted_relation()
                    if relation.parameter1 not in local_param_mapping:
                        if relation.parameter1_type not in local_additional_param_objects:
                            local_additional_param_objects[relation.parameter1_type] = []
                        if (
                            relation.parameter1
                            not in local_additional_param_objects[relation.parameter1_type]
                        ):
                            local_additional_param_objects[relation.parameter1_type].append(
                                relation.parameter1,
                            )

                    if relation.parameter2 not in local_param_mapping:
                        if relation.parameter2_type not in local_additional_param_objects:
                            local_additional_param_objects[relation.parameter2_type] = []
                        if (
                            relation.parameter2
                            not in local_additional_param_objects[relation.parameter2_type]
                        ):
                            local_additional_param_objects[relation.parameter2_type].append(
                                relation.parameter2,
                            )

            local_param_objects = set(local_param_mapping.keys())
            local_param_objects = LiftedPDDLAction.get_param_objects(
                local_param_objects,
                local_additional_param_objects,
            )
            for relation in state1.false_set:
                if relation not in local_changed:
                    lr = relation.get_lifted_relation()
                    if set([relation.parameter1, relation.parameter2]).issubset(
                        local_param_objects,
                    ):
                        if relation.parameter1 in local_param_mapping:
                            pid1 = local_param_mapping[relation.parameter1]
                        else:
                            if relation.parameter1 not in local_additional_param_mappings:
                                if relation.parameter1 in param_ids:
                                    pid1 = param_ids[relation.parameter1]
                                else:
                                    pid1 = (
                                        local_additional_param_objects[
                                            relation.parameter1_type
                                        ].index(relation.parameter1)
                                        + 1
                                    )
                                # local_additional_param_mappings[relation.parameter1] = relation.parameter1_type + "_" +  "extra" + "_p" + str(pid1)
                                local_additional_param_mappings[relation.parameter1] = (
                                    relation.parameter1_type + "_p" + str(pid1)
                                )
                            pid1 = local_additional_param_mappings[relation.parameter1]

                        if relation.parameter2 in local_param_mapping:
                            pid2 = local_param_mapping[relation.parameter2]
                        else:
                            if relation.parameter2 not in local_additional_param_mappings:
                                if relation.parameter2 in param_ids:
                                    pid2 = param_ids[relation.parameter2]
                                else:
                                    pid2 = (
                                        local_additional_param_objects[
                                            relation.parameter2_type
                                        ].index(relation.parameter2)
                                        + 1
                                    )
                                # local_additional_param_mappings[relation.parameter2] = relation.parameter2_type + "_" +  "extra" + "_p" + str(pid2)
                                local_additional_param_mappings[relation.parameter2] = (
                                    relation.parameter2_type + "_p" + str(pid2)
                                )
                            pid2 = local_additional_param_mappings[relation.parameter2]

                        parameterized_relation = LiftedRelation((pid1, pid2), lr)
                        relation_set.add(parameterized_relation)

            new_set = set()
            for relation in relation_set:
                for relation2 in neg_common_relations:
                    if relation == relation2:
                        new_set.add(relation)
                        break
            neg_common_relations = copy.deepcopy(new_set)

        ########## NEGATIVE PRECONDITION FINISHED

        param_set = set()
        for relation in common_relations:
            for param in relation.parameters:
                param_set.add(param)

        # ADD negative relations here
        for relation in neg_common_relations:
            for param in relation.parameters:
                param_set.add(param)

        for relation in cluster_e_add:
            for param in relation.parameters:
                param_set.add(param)

        for relation in cluster_e_delete:
            for param in relation.parameters:
                param_set.add(param)

        preconditions = Preconditions(
            true_set=common_relations,
            false_set=neg_common_relations,
        )

        effects = Effects(cluster_e_add, cluster_e_delete)

        LiftedPDDLAction.action_id += 1

        return LiftedPDDLAction(
            LiftedPDDLAction.action_id,
            sorted(param_set),
            preconditions,
            effects,
        )

    def __str__(self):
        s = f"(:action a{self.action_id} \n"
        param_string = ""
        for param in self.parameters:
            param_string += f" ?{param.pid} - {param.type} " if param.type else ""
        s += f":parameters ({param_string})\n"
        precondition_string = ""
        for i, param in enumerate(self.parameters):
            for j, param2 in enumerate(self.parameters):
                if j > i and param.type == param2.type:
                    precondition_string += f"\t(not (= ?{param.pid} ?{param2.pid}))\n"

        precondition_string += str(self.preconditions)

        s += f":precondition (and \n{precondition_string}) \n"

        effect_string = str(self.effects)

        s += f":effect (and \n {effect_string} ) \n"
        s += ")\n"

        return s

    def get_parameters(self):
        return [str(p) for p in self.parameters]


class RCR_bridge:
    def __init__(self, obj2pid: dict[str, int] = {}):
        self.obj2pid = obj2pid
        self.pid2type = {}
        self.obj2type = {}

    def operator_from_transitions(
        self,
        transition_tuples: list[list[PredicateState, PredicateState]],
        skill: Skill,
        type_dict: dict[str, list[str]],
        obj2type,
        flush=False,
    ) -> LiftedPDDLAction:
        """Convert PredicateState objects with grounded Predicate into PDDLState objects and build operators.
        obj2pid :: mapping of the original grounded parameters to ids of the lifted parameters
                    e.g., id of "object_p4" is 4
        """
        # For each transition, and each before/after state in the transition, for each predicate:
        #   Update the object set with the (presumably grounded) parameters of the predicate

        # For idx, object in the enumeration of the skill's concrete args:
        #   if object isn't in the mapping from object to parameter IDs:
        #       set the object ID for the object name to an incrementing integer
        #       set the parameter ID's type to the skill parameter's type, then increment obj_id

        # For objects in the object set:
        #   If object doesn't have a parameter ID assigned, give it an obj_id and increment obj_id

        # For all of the transition tuples, convert the PredicateState into PDDLState.
        #   TODO: This called self.predicatestate_to_pddlstate(t[0]) and on t[1]

        # For objects in the skill's params, map their PID to type to the object's type

        # Then compute the operator using LiftedPDDLAction.get_action_from_cluster(
        #     transition_cluster,
        #     copy.deepcopy(self.obj2pid),
        # )

    def get_pid_to_type(self) -> dict[int, str]:
        """Pid to type mapping is useful for generating possible groundings for precondition check."""
        # NOTE: One note on determining types when creating operators:
        #       If the parameter can only be in the skill, in the predicate, or in both.
        #       - If only in the skill, the type from the skill will be already the lowest hierarchy
        #       - If only in the predicate, we have handled finding lowest hierarchy before this
        #       - If in both, we use the predicates' type first
        pid2type = {}
        for obj, pid in self.obj2pid.items():
            if obj is None:
                pass
            elif obj in self.obj2type:
                pid2type[pid] = self.obj2type[obj]
            else:
                assert pid in self.pid2type, (
                    f"parameter must either be in the skill or the predicates of abstarct states, but {obj} is not in {list(self.pid2type.keys())}"
                )
                pid2type[pid] = self.pid2type[pid]
        return pid2type

    def unify_obj_type(
        self,
        transitions: list[list[PredicateState, PredicateState]],
        grounded_skill: Skill,
        type_dict,
        flush=False,
    ):
        """Find the common lowest hierarchy type of each parameters in the precondition and effect and the skill.
        by precauculate the precondition and effect and determine the lowest hierarchy type of each parameter.
        Ugly workaround since RCR code handles operator calculation and lifting at the same time

        Returns:
            unified_pddl_transitions :: original pddl transitions with parameter types replaced by common lowest hierarchy type

        """
        # build obj2pid mapping with skill parameter to be at the beginning (idx 0, 1)
        obj_set = set()
        for transition in transitions:
            for state in transition:
                for pred in state.iter_predicates():
                    obj_set.update(pred.params)

        obj_id = 0
        if flush:
            self.obj2pid = {}
        # params in the skill first
        for i, obj in enumerate(grounded_skill.params):
            if obj not in self.obj2pid:
                self.obj2pid[obj] = obj_id
                self.pid2type[obj_id] = grounded_skill.types[i]
                obj_id += 1

        # other params
        for obj in obj_set:
            if obj not in self.obj2pid:
                self.obj2pid[obj] = obj_id
                obj_id += 1

        pddl_transitions = [
            [self.predicatestate_to_pddlstate(t[0]), self.predicatestate_to_pddlstate(t[1])]
            for t in transitions
        ]

        # effect are the same across all transitions, so only need to calculate one
        eff_add = pddl_transitions[0][1].true_set - pddl_transitions[0][0].true_set
        eff_del = pddl_transitions[0][1].false_set - pddl_transitions[0][0].false_set

        # precondition is the intersection of all the init state
        precond_true = pddl_transitions[0][0].true_set
        precond_false = pddl_transitions[0][0].false_set
        for pddl_transition in pddl_transitions:
            precond_true &= pddl_transition[0].true_set
            precond_false &= pddl_transition[0].false_set

        # all parameters appeared in all *grounded relations* in precond and eff
        obj2type = {}
        for gr in precond_true | precond_false | eff_add | eff_del:
            if gr.p1.name is None:
                pass
            elif gr.p1.name not in obj2type or type_dict[gr.p1.name].index(gr.p1.type) > type_dict[
                gr.p1.name
            ].index(obj2type[gr.p1.name]):
                obj2type[gr.p1.name] = gr.p1.type

            if gr.p2.name is None:
                pass
            elif gr.p2.name not in obj2type or type_dict[gr.p2.name].index(gr.p2.type) > type_dict[
                gr.p2.name
            ].index(obj2type[gr.p2.name]):
                obj2type[gr.p2.name] = gr.p2.type

        # check if skill has even lower type hierarchy
        for obj, obj_type in zip(grounded_skill.params, grounded_skill.types, strict=False):
            if obj not in obj2type:
                obj2type[obj] = obj_type
            elif obj_type in type_dict[obj]:
                if type_dict[obj].index(obj_type) > type_dict[obj].index(obj2type[obj]):
                    obj2type[obj] = obj_type

        unified_pddl_transitions = copy.deepcopy(pddl_transitions)
        # replace all parameter types in all relations using obj2type
        for pddl_transition in unified_pddl_transitions:
            for pddl_state in pddl_transition:
                for grounded_relation in pddl_state.true_set:
                    grounded_relation.p1.type = (
                        obj2type[grounded_relation.p1.name]
                        if grounded_relation.p1.name is not None
                        else grounded_relation.p1.type
                    )
                    grounded_relation.parameter1_type = grounded_relation.p1.type
                    grounded_relation.p2.type = (
                        obj2type[grounded_relation.p2.name]
                        if grounded_relation.p2.name is not None
                        else grounded_relation.p2.type
                    )
                    grounded_relation.parameter2_type = grounded_relation.p2.type
                for grounded_relation in pddl_state.false_set:
                    grounded_relation.p1.type = (
                        obj2type[grounded_relation.p1.name]
                        if grounded_relation.p1.name is not None
                        else grounded_relation.p1.type
                    )
                    grounded_relation.parameter1_type = grounded_relation.p1.type
                    grounded_relation.p2.type = (
                        obj2type[grounded_relation.p2.name]
                        if grounded_relation.p2.name is not None
                        else grounded_relation.p2.type
                    )
                    grounded_relation.parameter2_type = grounded_relation.p2.type
        # breakpoint()
        return obj2type, unified_pddl_transitions


def generate_possible_groundings(pid2type, type_dict, fixed_grounding=None) -> list[dict[str, int]]:
    """required_types: list of types corresponding to total argument slots
    type_dict: dict of object -> type
    fixed_grounding: list of object names fixed at the beginning
    """
    required_types = [pid2type[i] for i in range(len(pid2type))]
    if fixed_grounding is None:
        fixed_grounding = []

    # Step 1: Validate fixed_grounding length
    if len(fixed_grounding) > len(required_types):
        raise ValueError("Fixed grounding has more objects than required types.")

    # Step 2: Remove fixed types and objects
    remaining_types = required_types[len(fixed_grounding) :]
    used_objects = set(fixed_grounding)

    # Step 3: Invert type_dict to type -> [objects]
    type_to_objects = {}
    for obj, tp_list in type_dict.items():
        for tp in tp_list:
            if obj not in used_objects:
                type_to_objects.setdefault(tp, []).append(obj)

    # Step 4: Gather object choices for remaining types
    try:
        object_choices = [type_to_objects[tp] for tp in remaining_types]
    except KeyError:
        # One of the remaining types has no available objects
        return []

    # Step 5: Generate combinations and filter duplicates
    groundings = []
    for combo in product(*object_choices):
        full_combo = tuple(fixed_grounding) + combo
        if len(set(full_combo)) == len(full_combo):
            obj2pid = {obj: i for obj, i in enumerate(full_combo)}
            groundings.append(obj2pid)

    return groundings


if __name__ == "__main__":
    test_pred_1 = Predicate("ClearAbove", ["pickupable"], ["PeanutButter"])
    test_pred_2 = Predicate("ClearAbove", ["pickupable"], ["Knife"])
    test_pred_3 = Predicate("Holding", ["robot", "pickupable"], ["Robot", "PeanutButter"])
    test_pred_4 = Predicate("LidRemoved", ["openable"], ["PeanutButter"])

    # # separate test
    bridge = RCR_bridge()
    test_ps_1 = PredicateState(
        [
            test_pred_1,
            test_pred_2,
            test_pred_3,
            test_pred_4,
        ],
    )
    test_ps_1.set_pred_value(test_pred_1, True)
    test_ps_1.set_pred_value(test_pred_2, False)
    test_ps_1.set_pred_value(test_pred_3, True)
    test_ps_1.set_pred_value(test_pred_4, True)
    test_ps_2 = copy.deepcopy(test_ps_1)

    # # skill = Skill("Open", ["openable"], ["PeanutButter"])
    # skill = Skill("Pick", ["utensil"], ["Knife"])
    skill = Skill("PickLeft", ["pickupable"], ["PeanutButter"])
    # skill = Skill("Scoop", ["pickupable", "pickupable"], ["Knife", "PeanutButter"])
    type_dict = {
        "PeanutButter": ["pickupable", "openable"],
        "Knife": ["pickupable", "utensil"],
        "Bread": ["food"],
        "Cup": ["receptacle"],
        "Table": ["location"],
        "Shelf": ["location"],
        "Robot": ["robot"],
    }
    transitions = [[test_ps_1, test_ps_2]]
    obj2type, _ = bridge.unify_obj_type(transitions, skill, type_dict)
    unified_transitions = []
    for t in transitions:
        unified_transition = []
        for state in t:
            predicate_state = PredicateState([])
            for grounded_pred, truth_value in state.pred_dict.items():
                types_list = []
                for idx, obj in enumerate(grounded_pred.params):
                    if obj in obj2type:
                        types_list.append(obj2type[obj])
                    else:
                        types_list.append(grounded_pred.types[idx])
                new_grounded_pred = Predicate(grounded_pred.name, types_list, grounded_pred.params)
                predicate_state.pred_dict[new_grounded_pred] = state.get_pred_value(grounded_pred)
            unified_transition.append(predicate_state)
        unified_transitions.append(unified_transition)

    operator = bridge.operator_from_transitions(
        unified_transitions,
        skill,
        type_dict,
        obj2type,
        flush=True,
    )
    breakpoint()
    # try grounding other objects using the lifted operator
    # test_pred_1 = Predicate("ClearAbove", ["openable"], ["PeanutButter"])
    # test_pred_2 = Predicate("ClearAbove", ["pickupable"], ["Knife"])
    # test_pred_3 = Predicate("Holding", ["robot", "openable"], ["Robot", "PeanutButter"])
    # test_pred_4 = Predicate("LidRemoved", ["openable"], ["PeanutButter"])

    # separate test
    # bridge = RCR_bridge()
    # test_ps_1 = PredicateState(
    #     [
    #         test_pred_1,
    #         test_pred_2,
    #         test_pred_3,
    #         test_pred_4,
    #     ]
    # )
    # test_ps_1.set_pred_value(test_pred_1, True)
    # test_ps_1.set_pred_value(test_pred_2, False)
    # test_ps_1.set_pred_value(test_pred_3, True)
    # test_ps_1.set_pred_value(test_pred_4, True)
    # test_ps_2 = copy.deepcopy(test_ps_1)
    # grounding = {0: "Knife", 1: "PeanutButter", 2: "Robot"}
    # pddl_state = bridge.predicatestate_to_pddlstate(test_ps_1, grounding)
    # param_name2param_object = {str(param): Parameter(param.pid, param.type, grounding[int(str(param).split("_p")[-1])]) for param in operator.parameters if not str(param).startswith("_")}
    # for param_name, param in param_name2param_object.items(): param_name2param_object[param_name].pid = str(param).split("_p")[-1]
    # param_name2param_object |= {'_p-1': Parameter(None, "", None)}
    # grounded_operator = operator.get_grounded_action(param_name2param_object, 0)
    # breakpoint()

    # another test
    test_pred_1 = Predicate("AboveSurface", ["pickupable"], ["PeanutButter"])
    test_pred_2 = Predicate("AboveSurface", ["pickupable"], ["Knife"])
    test_pred_3 = Predicate("EnclosedByGripper", ["robot", "pickupable"], ["Robot", "PeanutButter"])
    test_pred_4 = Predicate("EnclosedByGripper", ["robot", "pickupable"], ["Robot", "Knife"])
    test_pred_5 = Predicate("LidRemoved", ["openable"], ["PeanutButter"])
    bridge = RCR_bridge()
    test_ps_1 = PredicateState(
        [
            test_pred_1,
            test_pred_2,
            test_pred_3,
            test_pred_4,
            test_pred_5,
        ],
    )
    test_ps_1.set_pred_value(test_pred_1, True)
    test_ps_1.set_pred_value(test_pred_2, False)
    test_ps_1.set_pred_value(test_pred_3, False)
    test_ps_1.set_pred_value(test_pred_4, False)
    test_ps_1.set_pred_value(test_pred_5, False)

    test_ps_2 = copy.deepcopy(test_ps_1)
    test_ps_2.set_pred_value(test_pred_3, True)
    skill = Skill("PickLeft", ["pickupable"], ["PeanutButter"])
    transitions = [[test_ps_1, test_ps_2]]
    obj2type, _ = bridge.unify_obj_type(transitions, skill, type_dict)
    unified_transitions = []
    for t in transitions:
        unified_transition = []
        for state in t:
            predicate_state = PredicateState([])
            for grounded_pred, truth_value in state.pred_dict.items():
                types_list = []
                for idx, obj in enumerate(grounded_pred.params):
                    if obj in obj2type:
                        types_list.append(obj2type[obj])
                    else:
                        types_list.append(grounded_pred.types[idx])
                new_grounded_pred = Predicate(grounded_pred.name, types_list, grounded_pred.params)
                predicate_state.pred_dict[new_grounded_pred] = state.get_pred_value(grounded_pred)
            unified_transition.append(predicate_state)
        unified_transitions.append(unified_transition)
    operator = bridge.operator_from_transitions(
        unified_transitions,
        skill,
        type_dict,
        obj2type,
        flush=True,
    )
    breakpoint()
