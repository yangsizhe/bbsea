import os
import re
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
import sys 
sys.path.append("..") 
from utils import call_openai_api


with open(f'{os.path.dirname(os.path.abspath(__file__))}/prompt.txt','r',encoding='utf-8') as f:
    base_prompt = f.read()

def propose_task(scene_graph, temperature=0.2):
    prompt = base_prompt + '\n' + '```' + '\n' + 'scene graph:' + '\n' + scene_graph + 'tasks:' + '\n'
    # print('-----------------propose_task-------------------\n', prompt)

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
    response = call_openai_api(prompt, temperature=temperature)
    response = response['choices'][0]['message']['content']
    # tasks = response.split("\nanswer:\n")[-1].split("\n")  # str list
    # task_reasoning_list = []
    # task_desc_list = []
    # for task in tasks:
    #     task_reasoning = task.split(' |')[0].split('- ')[1]
    #     task_reasoning_list.append(task_reasoning)
    #     task_desc = task.split(' |')[1]
    #     task_desc_list.append(task_desc)

    tasks = response.split("\ntasks:\n")[-1].split("\n")  # str list
    task_desc_list = []
    for task in tasks:
        try:
            task_desc_list.append(task.split('- ')[1].strip('.'))
        except:
            try:
                task_desc_list.append(task.split('. ')[1].strip('.'))
            except:
                task_desc_list.append(task.split(': ')[1].strip('.'))
                print(tasks)
                print(task)

    return task_desc_list


if __name__ == '__main__':
    scene_graph = '''  [Nodes]:
    - red block
    - green bowl
    - table
  [Edges]:
    - red block -> on top of -> table
    - green bowl -> on top of -> table'''
    task_list = propose_task(scene_graph)
    print(task_list)
