import os
import re
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
import sys 
sys.path.append("..") 
from utils import call_openai_api
from robot_utils.primitive_actions import Pick, PlaceOn, PlaceAt, Push, PrismaticJointOpen, PrismaticJointClose, RevoluteJointOpen, RevoluteJointClose, Press
from math import * # sqrt, sin, cos, pi, tan, 

with open(f'{os.path.dirname(os.path.abspath(__file__))}/prompt.txt','r',encoding='utf-8') as f:
    base_prompt = f.read()
# is_first_time = True

def decompose_task(task_desc, scene_graph):
    prompt = base_prompt + '\n' + '```' + '\n' + 'task description: ' + task_desc + '\n' + 'scene graph:' + '\n' + scene_graph + 'reasoning: '
    # print('-----------------decompose_task-------------------\n', prompt)

    # # gpt3
    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=prompt,
    # )
    # response = response['choices'][0]['text']
    # gpt4
    # messages=[{"role": "user", "content": prompt}]
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=message,
    #     temperature=0.2,
    #     max_tokens=4000,
    #     frequency_penalty=0.0
    # )
    response = call_openai_api(prompt)
    response = response['choices'][0]['message']['content']
    subtask_list = []
    primitive_action_list = []
    primitive_action_str_list = []
    reasoning = response.split("\nanswer:\n")[0]
    if ' no.' in response:
        return subtask_list, primitive_action_str_list, primitive_action_list, reasoning
    steps = response.split("\nanswer:\n")[-1].split("\n")  # str list
    for step in steps:
        primitive_actions = []
        subtask = ''
        try:
            subtask = step.split('|')[0].split('.')[1]
            action = step.split('|')[1]
            # match = re.search(r'\[(.*)\]', step)
        
            match = re.search(r'\[(.*)\]', action)
            if match:
                primitive_action_str = match.group(1)
                primitive_action_str_list.append(primitive_action_str)
                
                for item in primitive_action_str.split(';'):
                    obj = eval(item) 
                    primitive_actions.append(obj)
            else:
                # import pdb;pdb.set_trace()
                return subtask_list, primitive_action_str_list, primitive_action_list, reasoning
        except:
            try:
                subtask = step.split(':')[0].split('.')[1]
                action = step.split(':')[1]
                # match = re.search(r'\[(.*)\]', step)

                match = re.search(r'\[(.*)\]', action)
                if match:
                    primitive_action_str = match.group(1)
                    primitive_action_str_list.append(primitive_action_str)

                    for item in primitive_action_str.split(';'):
                        obj = eval(item) 
                        primitive_actions.append(obj)
                else:
                    # import pdb;pdb.set_trace()
                    return subtask_list, primitive_action_str_list, primitive_action_list, reasoning
            except:
                print(step)
            # import pdb;pdb.set_trace()
        subtask_list.append(subtask)
        primitive_action_list.append(primitive_actions)
    # is_first_time = False
    return subtask_list, primitive_action_str_list, primitive_action_list, reasoning


if __name__ == '__main__':
    task_desc = 'put the red block in the green bowl'
    scene_graph = '''  [Nodes]:
    - red block
    - green bowl
    - table
  [Edges]:
    - red block -> on top of -> table
    - green bowl -> on top of -> table'''
    subtask_list, primitive_action_list = decompose_task(task_desc, scene_graph)
