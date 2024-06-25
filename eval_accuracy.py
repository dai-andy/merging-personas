import json
import numpy as np
import time
import torch
import re


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
            return 0, pred_answer
        pred_answer = most_frequent(pred_answers)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solutions)

    print("PRED", pred_answer)
    print("GT", gt)

    if gt == pred_answer:
        return 1, pred_answer
    else:
        return 0, pred_answer


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

if __name__ == "__main__":
    response_dict = torch.load('/content/drive/MyDrive/mind2mind/Finetuned_Debate.pt')
    questions = list(response_dict.keys())

    accuracies = []
    answers = []
    gts = []

    # compute and print accuracies
    for question in questions:
      responses, gt = response_dict[question]
      gts.append(gt)

      pred_solutions = []
      for response in responses:
          pred_solution = response[-1]['content']

          pred_solutions.append(pred_solution)

      accurate, pred_answer = compute_accuracy(gt, pred_solutions)

      answers.append(pred_answer)
      if accurate is not None:
          accuracies.append(float(accurate))
      else:
          import pdb
          pdb.set_trace()

      print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
