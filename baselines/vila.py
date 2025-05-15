# reproduction of vila baseline: given image of init state and final state and the history, return a next skill
import os
import sys
import argparse

sys.path.append("C:/Users/david/skillwrapper/skill_wrapper/src")

from utils import load_from_file, GPT4

def exec_vila(args):

    model = GPT4(engine=args.model)

    prompt = load_from_file("prompts/vila_prompt.txt")

    img_sequence = []
    if args.imgs_dir:
        # NOTE: closed-loop setting
        img_sequence = os.listdir(args.imgs_dir)
        prompt = prompt.replace("<style>", "the next possible action")
    else:
        # NOTE: open-loop setting
        img_sequence = [
            args.init_img,
            args.goal_img,
        ]
        prompt = prompt.replace("<style>", "a sequence of actions")

    metadata = load_from_file(f"task_config/{args.robot}.yaml")

    if args.robot == "dorfl":
        prompt = prompt.replace("<robot_description>", "a robot with two arms")
    elif args.robot == "spot":
        prompt = prompt.replace("<robot_description>", "a quadruped robot with a single arm")
    elif args.robot == "panda":
        prompt = prompt.replace("<robot_description>", "a single-armed robot mounted on a table")

    # -- let's formulate the prompt to include the skills and objects for the robot:
    skills = [str(metadata["skills"][P]) for P in metadata["skills"]]
    skills = [f"{sk+1}. {skills[sk]}" for sk in range(len(skills))]
    prompt = prompt.replace("<actions>", "\n".join(skills))

    objects = [f"- {O}: {metadata['objects'][O]['types']}" for O in metadata["objects"]]
    prompt = prompt.replace("<objects>", "\n".join(objects))

    # -- we will keep track of all actions proposed by
    interaction = []

    for x in range(len(img_sequence) - 1):
        # -- make a copy of the prompt string:
        new_prompt = str(prompt)

        # -- if there exists some history of actions, we will provide the
        if len(interaction):
            new_prompt += f" Your last set of actions were:\n"
            for y in range(len(interaction)):
                new_prompt += f"{y+1}. {interaction[y]}\n"

        resp = model.generate_multimodal(new_prompt, imgs=[img_sequence[x], img_sequence[x+1]])

        if "impossible" in resp[0].lower():
            print(resp)
            return []

        interaction.extend(resp[0].split('\n'))

    return interaction

#end

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot", type=str, default="dorfl", help="This specifies the robot being used: ['dorfl', 'spot', 'panda'].", )

    parser.add_argument("--init_img", type=str, default=None, help="This specifies the path to an image of the robot's INITIAL observation.", )
    parser.add_argument("--goal_img", type=str, default=None, help="This specifies the path to an image of the robot's FINAL observation.", )
    parser.add_argument("--imgs_dir", type=str, default=None, help="This specifies the path to a sequence of images for closed-loop planning.", )

    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", 'gpt-4o-2024-11-20'], default='gpt-4o-2024-11-20')

    args = parser.parse_args()

    plan = exec_vila(args)

    print(plan)