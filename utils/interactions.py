"""Utilities to faciliate interactions."""
import transformers
import re

import src

# For this prototype, just use the default sentiment analyser.
SENTIMENT_ANALYSER = transformers.pipeline("text-classification")


def generate_service_response(tutor: src.agents.Agent,
                              tutor_prompt: str,
                              tutor_prompt_parser_fn: callable,
                              validator: src.agents.Agent,
                              validator_prompt: str,
                              validator_prompt_parser_fn: callable,
                              assessment_: src.assessment.Assessment,
                              num_tries: int = 3
                              ) -> src.assessment.Assessment:
    """Generates a response for the writing service.

    Args:
        tutor: A tutor agent.
        tutor_prompt: A prompt for the tutor agent.
        tutor_prompt_parser_fn: A function that parses the tutor prompt.
        validator: A validator agent.
        validator_prompt: A prompt for the validator agent.
        validator_prompt_parser_fn: A function that parses the validator 
            prompt.
        assessment_: An assessment object.
        num_tries: The number of times to try to generate a response.

    Returns:
        An assessment object with the service's response.
    """
    response_is_valid = False
    tutor_output = None
    counter = 0

    while not response_is_valid and counter <= num_tries:
        # Prompt the tutor.
        tutor_output = tutor(
            prompt_template=tutor_prompt,
            assessment_=assessment_,
            prompt_parser_fn=tutor_prompt_parser_fn)

        # Prompt the validator.
        validator_output = validator(
            prompt_template=validator_prompt,
            assessment_=assessment_,
            prompt_parser_fn=validator_prompt_parser_fn,
            new_response=tutor_output)
        print(validator_output)

        # Get the sentiment of the validator's response.
        sentiment = SENTIMENT_ANALYSER(validator_output)[0]["label"]
        if sentiment == "POSITIVE":
            response_is_valid = True

        counter += 1

    # If total number of tries elapses, and the response is still not valid,
    # then return an error message.
    if not response_is_valid:
        tutor_output = ("I'm sorry, I'm having trouble responding. "
                        "Please try again later.")

    interaction = src.assessment.Interaction(service=tutor_output)
    assessment_.conversation.append(interaction)

    return assessment_


def clean_student_response(response: str) -> str:
    """Cleans the student's response.
    
    Args:
        response: The student's response.
    
    Returns:
        str: The cleaned response.
    """
    pattern = r'\b' + re.escape('tutor') + r'\b'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return response[:match.start()].strip()
    else:
        return response


def generate_student_response(student: src.agents.Agent,
                              student_prompt: str,
                              student_prompt_parser_fn: callable,
                              assessment_: src.assessment.Assessment
                              ) -> src.assessment.Assessment:
    """Generates a response for the student.

    Args:
        student: A student agent.
        student_prompt: A prompt for the student agent.
        student_prompt_parser_fn: A function that parses the student prompt.
        assessment_: An assessment object.
        num_tries: The number of times to try to generate a response.

    Returns:
        An assessment object with the student's response.
    """
    # Prompt the student.
    student_output = student(
        prompt_template=student_prompt,
        assessment_=assessment_,
        prompt_parser_fn=student_prompt_parser_fn)
    
    student_output = clean_student_response(student_output)
    student_output = student_output.replace(
        "Student:", "").replace("student:", "").strip()

    assessment_.conversation[-1].student = student_output

    return assessment_
