from pydantic import BaseModel, Field
from typing import List

class AnswerConclusion(BaseModel):
    """
    Represents the conclusion/result of an answer to a specific question,
    excluding reasoning or justification.
    """
    conclusion: str = Field(
        ...,
        description="The main conclusion or final decision expressed in the answer, without reasoning or justification."
    )

class AnswerReasoning(BaseModel):
    """
    Represents the reasoning or justification behind an answer as a list of individual points,
    excluding the final conclusion/result.
    """
    arguments: List[str] = Field(
        ...,
        description="A list of arguments/reasoning points or justifications extracted from the answer to support the answer's conclusion. Excludes the final conclusion itself. Only extract arguments that are explicitly contained in the answer."
    )

class ArgumentSupported(BaseModel):
    """
    Represents whether an argument is supported by a list of other arguments,
    based on semantic equivalence or clear implication.
    """
    supported: bool = Field(
        ...,
        description="True if the argument is supported by at least one of the reference arguments, False otherwise."
    )