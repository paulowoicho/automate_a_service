"""Entry point for the application."""
import datetime
import pathlib

import torch

import src
import utils

# Define the agents.
student = src.agents.Agent(utils.inference.model_fn)
tutor = src.agents.Agent(utils.inference.model_fn)
validator = src.agents.Agent(utils.inference.model_fn)

# Create an empty assessment. The topic can be anything, but the standard must
# be a valid Common Core standard. However, validation is out of scope for
# this prototype.
assessment_ = src.assessment.Assessment(
    common_core_standard="Draw evidence from literary or informational texts to support analysis, reflection, and research.",
    topic_of_interest="Baseball"
)

# Define the prompt templates.
tutor_prompt = utils.inference.load_prompt(
    pathlib.Path(__file__).parent.resolve() / "utils/prompt_templates/llama2_tutor_prompt.txt")

student_prompt = utils.inference.load_prompt(
    pathlib.Path(__file__).parent.resolve() / "utils/prompt_templates/llama2_student_prompt.txt")

validator_prompt = utils.inference.load_prompt(
    pathlib.Path(__file__).parent.resolve() / "utils/prompt_templates/llama2_validator_prompt.txt")

# Define the number of interactions before the conversations ends.
NUM_INTERACTIONS = 5

while len(assessment_.conversation) <= NUM_INTERACTIONS:
    assessment_ = utils.interactions.generate_service_response(
        tutor=tutor,
        tutor_prompt=tutor_prompt,
        tutor_prompt_parser_fn=src.agents.tutor_prompt_parser_fn,
        validator=validator,
        validator_prompt=validator_prompt,
        validator_prompt_parser_fn=src.agents.validator_prompt_parser_fn,
        assessment_=assessment_)
    
    assessment_ = utils.interactions.generate_student_response(
        student=student,
        student_prompt=student_prompt,
        student_prompt_parser_fn=src.agents.student_prompt_parser_fn,
        assessment_=assessment_)

# Empty the GPU cache.
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Save the conversation.
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"transcripts/conversation_{current_datetime}.txt"
# Create the transcripts directory if it doesn't exist.
pathlib.Path("transcripts").mkdir(parents=True, exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    for interaction in assessment_.conversation:
        f.write(f"Writing Service:\n{interaction.service}")
        f.write("\n\n")
        f.write(f"Student:\n{interaction.student}")
        f.write("\n\n")
