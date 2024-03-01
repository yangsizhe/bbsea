import os
import re
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
import sys 
sys.path.append("..") 
from utils import call_openai_api

with open(f'{os.path.dirname(os.path.abspath(__file__))}/prompt.txt','r',encoding='utf-8') as f:
    base_prompt = f.read()

def infer_if_success(task_desc, scene_graph_list):
    scene_graph_list_str = ''
    for scene_graph in scene_graph_list:
        scene_graph_list_str += f'  ----------\n{scene_graph}'
    prompt = base_prompt + '\n' + '```' + '\n' + 'task description: ' + task_desc + '\n' + 'scene graph list:' + '\n' + scene_graph_list_str + 'success metric: '
    # print('-----------------infer_if_success-------------------\n', prompt)

    # # gpt3
    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=prompt,
    # )
    # response = response['choices'][0]['text']
    # # gpt4
    # message=[{"role": "user", "content": prompt}]
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=message,
    #     temperature=0.2,
    #     max_tokens=4000,
    #     frequency_penalty=0.0
    # )
    response = call_openai_api(prompt)
    response = response['choices'][0]['message']['content']
    answer = response.split('answers:\n')[-1]

    if 'yes' in answer:
        return True, response
    elif 'no' in answer:
        return False, response
    else:
        return 'not sure', response


if __name__ == '__main__':
    task_desc = 'put the red block in the green bowl'
    scene_graph = '''  [Nodes]:
    - red block
    - green bowl
    - table
  [Edges]:'''
    is_successful = infer_if_success(task_desc, scene_graph)
    print(is_successful)