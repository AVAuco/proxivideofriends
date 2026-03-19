"""
Utility functions for prompt generation, response parsing, and mAP computation.
"""

import re
import numpy as np
from sklearn.metrics import average_precision_score


# -----------------------------
# Prompt generation
# -----------------------------
def generate_prompt(idx_question, questions_json):
    """
    Build the prompt for a given question index.

    Args:
        idx_question (int): Index of the question to use.
        questions_json (dict): Dictionary containing prompt components.

    Returns:
        str: Full prompt for the selected question.
    """
    prompt = questions_json["physical_interaction_definition"][idx_question]
    prompt += questions_json["questions_proxemics"][idx_question]
    prompt += questions_json["answer_choices_proxemics"]
    prompt += questions_json["output_proxemics"]
    prompt += " Do not add any explanation, prefix, suffix, or commentary."
    return prompt


# -----------------------------
# Response parsing
# -----------------------------
def parse_response_to_vector(response, num_classes=6):
    """
    Extract predicted answers from the model output and convert them
    into a multi-hot vector.

    The function looks for answers in the format:
        - A
        - A, B
        - Answer: A
        - the correct answer is: A, C

    Args:
        response (str): Raw model output.
        num_classes (int, optional): Number of output classes. Defaults to 6.

    Returns:
        list: Multi-hot prediction vector.
    """
    # Regular expression used to capture one or multiple answer letters
    pattern = r"(?:Answer:\s*|the correct answer is:\s*)?\(?([A-G](?:,\s*[A-G])*)\)?"

    match = re.search(pattern, response)

    if not match:
        return [0] * num_classes
    # Split the matched answers and remove extra spaces
    answers = match.group(1).replace(" ", "").split(',')
    # Map answer letters to class indices
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    vector = [0] * num_classes

    for a in answers:
        if a in mapping:
            vector[mapping[a]] = 1

    return vector



# -----------------------------
# mAP computation
# -----------------------------
def compute_map(real_labels, all_predictions):
    """
    Compute Average Precision (AP) per class and mean Average Precision (mAP)
    for each question.

    Args:
        real_labels (list): Ground-truth labels.
        all_predictions (list): Predicted multi-hot vectors for each question.

    Returns:
        dict: Dictionary containing AP per class and mAP per question.
    """
    result_json = {}
    real_labels = np.array(real_labels)

    for q_idx, preds in enumerate(all_predictions):
        preds = np.array(preds)

        AP = []

        for class_idx in range(preds.shape[1]):
            y_true = real_labels[:, class_idx]
            y_pred = preds[:, class_idx]

            if not np.any(y_true):
                AP.append(np.nan)
                continue

            if np.all(y_pred == 0):
                AP.append(0.0)
                continue

            AP.append(average_precision_score(y_true, y_pred))

        result_json[f"Question {q_idx + 1}"] = {
            "AP per class": AP,
            "mAP": float(np.mean(AP))
        }

    return result_json