"""
Use priviledged predicates from burger simulator to learn operators.
Propose skill sequence as SkillWrapper, get predicate state directly without classification.
"""

# read plan from results, run the plans and save the predicate states in yaml files under results/transitions

# Call operator learning function in operator_learning.py to learn operators
import argparse
import logging
from collections import defaultdict

from src.utils import save_to_file, load_from_file, setup_logging
from src.invent_predicate import filter_predicates, calculate_operators_for_all_skill
from src.skill_sequence_proposing import SkillSequenceProposing


def propose_and_execute(skill_sequence_proposing: SkillSequenceProposing, tasks, lifted_pred_list, skill2operator, args):
    pass

def main():
    pass