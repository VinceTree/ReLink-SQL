"""Evaluation-side simulation of an accurate binary user; never builds prompts."""
import json
from pathlib import Path

from recoverylab.sql import result_equal

NEGATIVE = 'The result is incorrect. Please correct it.'


def response(accepted):
    if type(accepted) is not bool:
        raise ValueError('Feedback requires a scorable Boolean acceptance')
    return dict(accepted=accepted, feedback='' if accepted else NEGATIVE)


class UserSimulator:
    def __init__(self, reference_path):
        self._references = json.loads(Path(reference_path).read_text())

    def assess(self, task_id, observation):
        reference = self._references[task_id]
        if reference['gold_result']['status'] != 'ok':
            raise ValueError('Unscorable reference: ' + task_id)
        return response(bool(result_equal(observation, reference['gold_result'],
                                          reference['order_matters'])))
