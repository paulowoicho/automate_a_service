"""Classes and functions for LLM Based Agents."""
from typing import Callable, Dict

from . import assessment


def tutor_prompt_parser_fn(prompt_template: str,
                           assessment_: assessment.Assessment) -> str:
    """Parses the tutor prompt template.

    Args:
        prompt_template: A string template for the prompt with fields that need
            to be populated.
        assessment_: An Assessment object.

    Returns:
        str: The parsed prompt.
    """
    transcript = ""
    for interaction in assessment_.conversation:
        if interaction.service:
            transcript += f"{interaction.service} </s><s>[INST] "
        if interaction.student:
            transcript += f"{interaction.student} [\INST]"

    prompt = prompt_template.format(
        common_core_standard=assessment_.common_core_standard,
        topic=assessment_.topic_of_interest,
        interaction=transcript
    ).strip()

    return prompt


def student_prompt_parser_fn(prompt_template: str,
                             assessment_: assessment.Assessment) -> str:
    """Parses the student prompt template.

    Args:
        prompt_template: A string template for the prompt with fields that need
            to be populated.
        assessment_: An Assessment object.

    Returns:
        str: The parsed prompt.
    """
    transcript = ""
    for interaction in assessment_.conversation:
        if interaction.service:
            transcript += f"Tutor: {interaction.service} [\INST] "
        if interaction.student:
            transcript += f"Student: {interaction.student} </s><s>[INST] "

    prompt = prompt_template.format(interaction=transcript).strip()

    return prompt


def validator_prompt_parser_fn(prompt_template: str,
                               assessment_: assessment.Assessment,
                               **kwargs) -> str:
    """Parses the validator prompt template.

    Args:
        prompt_template: A string template for the prompt with fields that need
            to be populated.
        assessment_: An Assessment object.
        **kwargs: Optional arguments to be passed to the prompt_parser_fn.

    Returns:
        str: The parsed prompt.
    """
    transcript = ""
    for interaction in assessment_.conversation:
        if interaction.service:
            transcript += f"Tutor: {interaction.service}\n"
        if interaction.student:
            transcript += f"Student: {interaction.student}\n"

    prompt = prompt_template.format(
        interaction=transcript.strip(),
        new_response=kwargs["new_response"]
    )

    return prompt


class Agent:
    """Defines an agent that can be used as a student or the subcomponents
    of a teaching service."""

    def __init__(self, model_fn: Callable[[str], str]):
        """Initializes the agent.

        Args:
            model_fn: A function that takes in a prompt and returns a response.
        """
        self.model_fn = model_fn

    def __call__(self, *args, **kwargs) -> str:
        return self.run(*args, **kwargs)

    def run(self,
            prompt_template: str,
            assessment_: assessment.Assessment,
            prompt_parser_fn: Callable[[str, assessment.Assessment, Dict], str],
            **kwargs) -> str:
        """Runs the agent on the given prompt and assessment.

        Args:
            prompt_template: A string template for the prompt. May contain 
                fields that need to be populated by the prompt_parser_fn.
            assessment_: An Assessment object.
            prompt_parser_fn: A function that takes in a prompt template,
                an assessment, and optional arguments and returns a string.
            **kwargs: Optional arguments to be passed to the prompt_parser_fn.
            """
        prompt = prompt_parser_fn(prompt_template, assessment_, **kwargs)
        return self.model_fn(prompt)
