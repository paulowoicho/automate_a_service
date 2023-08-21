"""Utilities for inference."""

import functools
import os
import pathlib
from typing import Callable

import dotenv
import huggingface_hub
import torch
import transformers

# Log into the Hugging Face Hub
dotenv.load_dotenv()
huggingface_hub.login(os.getenv("HUGGINGFACE_TOKEN"))

# Define the model function. This function will be passed to the Agent class.
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# Hyperparameters for the model.
MAX_NEW_TOKENS = 500
NUM_RETURN_SEQUENCES = 1
TEMPERATURE = 1.0
ROPES_SCALING_TYPE = "dynamic"
ROPES_SCALING_FACTOR = 2.0

tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
pipeline = transformers.pipeline(
    "text-generation",
    model=BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    # For long contexts, we need to set rope_scaling params.
    model_kwargs={"rope_scaling": {
        "type": ROPES_SCALING_TYPE, "factor": ROPES_SCALING_FACTOR}}
)

pipeline_stub = functools.partial(
    pipeline, do_sample=True, top_k=10,
    num_return_sequences=NUM_RETURN_SEQUENCES,
    eos_token_id=tokenizer.eos_token_id,
    temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
)


def model_fn(text: str, fn_stub: Callable[[str], str] = pipeline_stub) -> str:
    """A function that takes a text and returns a response.

    Args:
        text (str): The text to respond to.
        fn_stub (Callable[[str], str]): The function that will be used to 
            respond to the text.

    Returns:
        str: The response to the text.
    """
    generated_text = fn_stub(text)[0]['generated_text']
    # Remove the text that was used to generate the response.
    generated_text = generated_text.replace(text, '').strip()
    return generated_text


def load_prompt(path: pathlib.Path) -> str:
    """Loads a prompt from a file.

    Args:
        path (pathlib.Path): The path to the prompt file."""
    with open(path, encoding="utf-8") as f:
        return f.read()
