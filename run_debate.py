import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from glob import glob
import pandas as pd
import json
import time
import random
import os
import numpy as np
from tqdm import tqdm
import torch

# set  PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

# change personas and other_persona according to experiment
personas = [
'You are a humanities professor who has a deep understanding of human culture, history, philosophy, and the arts. ',
'You are a mathematician who has strong quantitative skills and who provides analytical and logical perspectives, often using mathematical principles and models to address questions. ',
'You are a doctor who provides medical and health-related expertise, focusing on the biological, psychological, and physiological aspects of issues. '
]

other_persona = [
"This is an answer from a humanities professor who has a deep understanding of human culture, history, philosophy, and the arts. ",
"This is an answer from a mathematician who has strong quantitative skills and who provides analytical and logical perspectives, often using mathematical principles and models to address questions. ",
"This is an answer from a doctor who provides medical and health-related expertise, focusing on the biological, psychological, and physiological aspects of issues. "
]

model_names = ['daiandy/humanities-model', 'daiandy/math-model', 'daiandy/doctor-model']


def construct_message(a_i, agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Pick A, B, C, or D and put your answer in the form (A), (B), (C), or (D) at the end of your response."}

    pre_role = personas[a_i]
    prefix_string = pre_role + "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string += response


    prefix_string += """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Pick A, B, C, or D and put your answer in the form (A), (B), (C), or (D) at the end of your response."""
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(a_i, completion):
    content = completion[0]['generated_text'][1]['content']
    prefix = other_persona[a_i]
    return {"role": "assistant", "content": prefix + content}


if __name__ == "__main__":
    token = os.getenv("HUGGINGFACE_TOKEN")

    # use three gpus
    device_ids = [0, 1, 2]
    print(f"Using GPUs: {device_ids}")

    # load tokenizers and models on specific GPUs
    tokenizers = [AutoTokenizer.from_pretrained(model_name, use_auth_token=token) for model_name in model_names]
    models = [
        AutoModelForCausalLM.from_pretrained(model_names[0], use_auth_token=token).to('cuda:0'),
        AutoModelForCausalLM.from_pretrained(model_names[1], use_auth_token=token).to('cuda:1'),
        AutoModelForCausalLM.from_pretrained(model_names[2], use_auth_token=token).to('cuda:2')
    ]

    pipelines = [
        pipeline("text-generation", model=models[0], tokenizer=tokenizers[0], device=0, max_new_tokens=7000),
        pipeline("text-generation", model=models[1], tokenizer=tokenizers[1], device=1, max_new_tokens=7000),
        pipeline("text-generation", model=models[2], tokenizer=tokenizers[2], device=2, max_new_tokens=7000)
    ]

    agents = 3
    rounds = 3

    dataset = {}

    loaded_dict = torch.load('GT.pt')

    # run debate for each MMLU prompt
    for question, answer in tqdm(loaded_dict.items(), desc="Processing"):
        agent_contexts = [[{"role": "user", "content": personas[agent] + question}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(i, agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)


                completion = pipelines[i](agent_context)
                assistant_message = construct_assistant_message(i, completion)
                agent_context.append(assistant_message)


                print(f"ROUND: {round}, AGENT: {i}")
                print(agent_context)


        dataset[question] = (agent_contexts, answer)

    torch.save(dataset, 'Finetuned_Debate.pt')