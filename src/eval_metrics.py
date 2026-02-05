from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase

def g_eval_conclusion_alignment(question, human_answer, ai_answer, eval_model):
    alignment_metric = GEval(
        name="Human-AI Conclusion Alignment",
        criteria="""
            Evaluate whether the AI response (actual output) and human response (expected output) reach the same main conclusion or decision regarding a question.
            - Focus on the final choice, preference, or outcome expressed. Do not consider the reasoning or things that influence the decision. Only consider the final outcome. 
            - The AI does not need to match wording exactly, but should express the same conclusion.
            - Assign a score for conclusion alignment on a scale of 0 to 1, where 0 is the lowest (conclusions are different) and 1 is the highest (conclusions are the same).
            - Sometimes, conclusions can be partially the same, which should be reflected with an appropriate score between 0 and 1.
            """,
        model=eval_model,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=ai_answer,
        expected_output=human_answer
    )

    alignment_metric.measure(test_case)

    return alignment_metric