import requests
from dataclasses import dataclass
from typing import List


@dataclass
class NominalFrame:
    nominal: str
    sense: str
    predicate_index: List[int]
    description: str
    tags: List[str]


@dataclass
class NomSRLResult:
    nominals: List[NominalFrame]
    words: List[str]


class WebNomSRLPredictor:
    def __init__(self, host='127.0.0.1', port=8984):
        self.url = f'http://{host}:{port}/cogcomp_nom_srl'
        self.session = requests.Session()

    @staticmethod
    def sanitize(json_dict: dict):
        return NomSRLResult(
            nominals=[NominalFrame(**d) for d in json_dict['nominals']],
            words=json_dict['words'],
        )

    def predict(self, sentence: str) -> NomSRLResult:
        return self.sanitize(self.session.post(self.url, json={'sentence': sentence}).json())

    def batch_predict(self, sentences: List[str]) -> List[NomSRLResult]:
        res_json = self.session.post(self.url, json=[{'sentence': s} for s in sentences]).json()
        return [self.sanitize(dic) for dic in res_json]


@dataclass
class VerbFrame:
    verb: str
    sense: str
    description: str
    tags: List[str]


@dataclass
class VerbSRLResult:
    verbs: List[VerbFrame]
    words: List[str]


class WebVerbSRLPredictor:
    def __init__(self, host='127.0.0.1', port=8983):
        self.url = f'http://{host}:{port}/cogcomp_verb_srl'
        self.session = requests.Session()

    @staticmethod
    def sanitize(json_dict: dict):
        return VerbSRLResult(
            verbs=[VerbFrame(**d) for d in json_dict['verbs']],
            words=json_dict['words'],
        )

    def predict(self, sentence: str) -> VerbSRLResult:
        return self.sanitize(self.session.post(self.url, json={'sentence': sentence}).json())

    def batch_predict(self, sentences: List[str]) -> List[VerbSRLResult]:
        res_json = self.session.post(self.url, json=[{'sentence': s} for s in sentences]).json()
        return [self.sanitize(dic) for dic in res_json]
