import pandas as pd
import os
from src.custom_llm import *
from src.eval_metrics import *
from src.pydantic_models import *
from src.prompts import *

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "gpt-5-mini-2025-08-07"
BASE_URL = "https://api.openai.com/v1"
INPUT_PATH = "../data/RB_GenAI_Datatest.xlsx"
OUTPUT_PATH = "../data/RB_assignment_processed.csv"


# ---------------------------------------------------------------------
# Row-level computation functions
# ---------------------------------------------------------------------

def compute_conclusion_row(row, client):
    """
    Extract human and AI conclusions for a row and compute alignment score.
    """

    # Extract AI conclusion
    ai_conclusion = call_llm_with_retries(
        client=client,
        llm_name=MODEL_NAME,
        schema=AnswerConclusion,
        user_prompt=extract_conclusion_user_prompt(
            question=row["question"],
            answer=row["ai_answers"],
        ),
        system_prompt=extract_conclusion_system_prompt(),
        temperature=1,
    )

    # Extract Human conclusion
    human_conclusion = call_llm_with_retries(
        client=client,
        llm_name=MODEL_NAME,
        schema=AnswerConclusion,
        user_prompt=extract_conclusion_user_prompt(
            question=row["question"],
            answer=row["human_answers"],
        ),
        system_prompt=extract_conclusion_system_prompt(),
        temperature=1,
    )

    # Compute alignment score
    score = g_eval_conclusion_alignment(
        question=row["question"],
        human_answer=human_conclusion.conclusion,
        ai_answer=ai_conclusion.conclusion,
        eval_model=MODEL_NAME,
    )

    return pd.Series(
        {
            "human_conclusion": human_conclusion.conclusion,
            "ai_conclusion": ai_conclusion.conclusion,
            "conclusion_score": score.score,
            "conclusion_reason": score.reason,
        }
    )


def get_arguments_row(row, client):
    """
    Extract human and AI arguments and compute precision, recall, and F1.
    """

    # Extract AI arguments
    ai_arguments = call_llm_with_retries(
        client=client,
        llm_name=MODEL_NAME,
        schema=AnswerReasoning,
        user_prompt=extract_reasoning_user_prompt(
            question=row["question"],
            answer=row["ai_answers"],
        ),
        system_prompt=extract_reasoning_system_prompt(),
        temperature=1,
    )

    # Extract Human arguments
    human_arguments = call_llm_with_retries(
        client=client,
        llm_name=MODEL_NAME,
        schema=AnswerReasoning,
        user_prompt=extract_reasoning_user_prompt(
            question=row["question"],
            answer=row["human_answers"],
        ),
        system_prompt=extract_reasoning_system_prompt(),
        temperature=1,
    )

    # Check AI arguments supported by human arguments
    ai_arguments_supported = []
    for arg in ai_arguments.arguments:
        result = call_llm_with_retries(
            client=client,
            llm_name=MODEL_NAME,
            schema=ArgumentSupported,
            user_prompt=check_argument_support_user_prompt(
                ai_argument=arg,
                human_arguments=human_arguments.arguments,
            ),
            system_prompt=check_argument_support_system_prompt(),
            temperature=1,
        )
        ai_arguments_supported.append(result.supported)

    # Check human arguments supported by AI arguments
    human_arguments_supported = []
    for arg in human_arguments.arguments:
        result = call_llm_with_retries(
            client=client,
            llm_name=MODEL_NAME,
            schema=ArgumentSupported,
            user_prompt=check_argument_support_user_prompt(
                ai_argument=arg,
                human_arguments=ai_arguments.arguments,
            ),
            system_prompt=check_argument_support_system_prompt(),
            temperature=1,
        )
        human_arguments_supported.append(result.supported)

    num_ai_args = len(ai_arguments.arguments)
    num_human_args = len(human_arguments.arguments)

    # Precision
    if num_ai_args == 0:
        argument_precision = 1.0 if num_human_args == 0 else 0.0
    else:
        argument_precision = sum(ai_arguments_supported) / num_ai_args

    # Recall
    if num_human_args == 0:
        argument_recall = 1.0
    else:
        argument_recall = sum(human_arguments_supported) / num_human_args

    # F1 score
    if argument_precision + argument_recall == 0:
        argument_f1 = 0.0
    else:
        argument_f1 = (
            2 * argument_precision * argument_recall
            / (argument_precision + argument_recall)
        )

    return pd.Series(
        {
            "human_arguments": human_arguments.arguments,
            "ai_arguments": ai_arguments.arguments,
            "argument_precision": argument_precision,
            "argument_recall": argument_recall,
            "argument_f1": argument_f1,
        }
    )


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main():
    """
    Main entry point for the script.
    """

    # Load API key
    os.environ["OPENAI_API_KEY"] = "..."
    client = LLMClient(os.environ["OPENAI_API_KEY"], BASE_URL)

    # Load data
    df = pd.read_excel(INPUT_PATH)

    # Compute conclusion-level metrics
    df[
        [
            "human_conclusion",
            "ai_conclusion",
            "conclusion_score",
            "conclusion_reason",
        ]
    ] = df.apply(lambda row: compute_conclusion_row(row, client), axis=1)

    # Compute argument-level metrics
    df[
        [
            "human_arguments",
            "ai_arguments",
            "argument_precision",
            "argument_recall",
            "argument_f1",
        ]
    ] = df.apply(lambda row: get_arguments_row(row, client), axis=1)

    # Save results
    df.to_csv(OUTPUT_PATH, sep=";", index=False)


if __name__ == "__main__":
    main()