import json
import openai
import numpy as np
import time
import torch
import re
import random
import os
import pandas as pd
from tqdm import tqdm


def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


# Function to escape regex special characters
def escape_special_chars(s):
    return re.escape(s)


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None


def parse_answer(input_str):
    pattern = r'\((\w)\)'
    matches = re.findall(pattern, input_str)

    solution = None
    # print("predicted solution")
    # print(input_str)
    # print("matches")
    # print(matches)

    options = ["A", "B", "C", "D"]
    ans_dict = {"A":0, "B":1, "C":2, "D":3}

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution in options:
          #solution = ans_dict[solution]
          break

    #print("SOLUTION", solution)
    #print(input_str)
    #print("AFTER PARSING", solution)
    return solution


def compute_accuracy(gt, pred_solutions):
    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if pred_answer is None:
            return 0
        pred_answer = most_frequent(pred_answers)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solutions)

    if gt == pred_answer:
        return 1
    else:
        return 0


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


def is_correct(gt, response):
  pred_solution = response
  accurate = compute_accuracy(gt, pred_solution)

  return accurate


def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer. Pick A, B, C, or D and put your answer in the form (A), (B), (C), or (D) at the end of your response.".format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer


personas = ['You are a humanities professor who has a deep understanding of human culture, history, philosophy, and the arts. ',
           'You are a mathematician who has strong quantitative skills and who provides analytical and logical perspectives, often using mathematical principles and models to address questions. ',
           'You are a doctor who provides medical and health-related expertise, focusing on the biological, psychological, and physiological aspects of issues. ']


# START GENERATION
file_name = '/content/drive/MyDrive/mind2mind/baseline/mmlu/mmlu.csv'

tasks = glob(file_name + '.csv')

dfs = [pd.read_csv(task) for task in tasks]

random.seed(0)
q_num = 0

# EACH QUESTION
dataset = {}

for i in tqdm(range(1000)):

  question_repeat = True
  question, answer = None, None

  while question_repeat:
    ix = len(df)
    idx = random.randint(0, ix-1)
    question, answer = parse_question_answer(df, idx)
    escaped_question = escape_special_chars(question)
    question_repeat = df['prompt'].str.contains(escaped_question, case=False, na=False).any()

  agent_context = [{"role": "user", "content": personas[0] + question}]

  completion = client.chat.completions.create(
                  model="meta-llama/Llama-3-70b-chat-hf",
                  messages=agent_context,
              )
  completion = completion.choices[0].message.content

  accurate = is_correct(answer, completion)
  print(accurate)

  if not accurate:
    msg = "THIS-ANSWER-IS-WRONG"
    completion = msg + completion
    print(msg)

  dataset[question] = completion
