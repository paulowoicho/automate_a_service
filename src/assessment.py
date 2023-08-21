"""Defines the Assessment class.

Assessment is a class that represents an interaction between a student and a
teaching service. It is used to track the conversation that takes place between
the student and the service, including the initial free response question, 
the student's response, the service's feedback, and other subsequent
interactions. 

The Assessment class also contains the student's topic of interest and the
relevant Common Core standard that the service is assessing.
"""
import dataclasses
from typing import List


@dataclasses.dataclass
class Interaction:
    """Defines an interaction between a student and a teaching service.

    An interaction can be thought of as a turn in a conversation. It holds
    the text that was sent by the service to the student and the text that
    the student sends back.
    """
    service: str = None
    student: str = None


@dataclasses.dataclass
class Assessment:
    """Defines an assessment of a student's response to a free response question.

    An assessment is a collection of interactions between a student and a
    teaching service. It holds the student's response to a free response
    question, the service's feedback, and other subsequent interactions.
    """
    common_core_standard: str
    topic_of_interest: str
    conversation: List[Interaction] = dataclasses.field(default_factory=list)
