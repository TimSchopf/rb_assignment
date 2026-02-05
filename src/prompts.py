def extract_conclusion_system_prompt() -> str:
    return "You are an expert in extracting the main conclusion of an answer to a specific question."


def extract_conclusion_user_prompt(question: str, answer: str) -> str:
    return f'''Please extract only the main conclusion/result of the answer to the question **without** the reasoning/justification.
    
    Question: where do you typically purchase your personal care products, and why?
    Answer: i typically purchase my personal care products from stores that offer a wide range of essentials, like supermarkets or online platforms this way, i can get everything i need in one place, saving time and effort i also look for retailers that provide good value for money and carry products with non-harmful ingredients.
    Conclusion: Typically from stores.
    
    Question: where do you typically purchase your personal care products, and why?
    Answer: i usually buy my products online, as this gives the most flexibility and there are some nice discouns. sometimes i buy them at stores when i see something interesting.
    Conclusion: usually online but sometimes at stores.
    
    Question: {question}
    Answer: {answer}
    Conclusion: 
    '''

def extract_reasoning_system_prompt() -> str:
    return "You are an expert in extracting the reasoning or justification behind an answer, without including the final conclusion/result. Extract each reasoning point separately."


def extract_reasoning_user_prompt(question: str, answer: str) -> str:
    return f'''Please extract only the arguments of the reasoning/justification from the answer to the question, **without** including the final conclusion. List each point as a separate item.
    
    Question: {question}
    Answer: {answer}
    Arguments: 
    '''

def check_argument_support_system_prompt() -> str:
    return (
        "You are an expert in comparing reasoning arguments. "
        "Your task is to determine whether an AI argument is supported by, "
        "contained in, or semantically equivalent to any of the human arguments."
    )


def check_argument_support_user_prompt(
    ai_argument: str,
    human_arguments: list[str],
) -> str:
    return f'''Determine whether the AI argument is supported by the list of human arguments.

    An AI argument is considered SUPPORTED if:
    - It expresses the same idea as at least one human argument, OR
    - It is a clear paraphrase or more general/specific version of a human argument.
    
    It is NOT SUPPORTED if:
    - The idea is missing from the human arguments
    - The idea contradicts the human arguments
    - The idea introduces a new justification not mentioned by the human
    
    Respond with:
    Supported: True (yes) or False (no)

    
    AI argument: {ai_argument}
    Human arguments:
    {chr(10).join(f"- {arg}" for arg in human_arguments)}
    
    Supported:
'''
