import unittest
from dataclasses import asdict
from clients import WebNomSRLPredictor, NomSRLResult


class VerbSRLTestcase(unittest.TestCase):
    sentences = ['Twitter confirms sale of company to Elon Musk for $44 billion.']
    references = [
        {
            'nominals': [{'nominal': 'sale', 'sense': '01', 'predicate_index': [2],
                          'description': '[ARG0: Twitter] [Support: confirms] sale [ARG1: of company] [ARG2: to Elon Musk] [ARG3: for $ 44 billion] .',
                          'tags': ['B-ARG0', 'B-Support', 'O', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2',
                                   'I-ARG2', 'B-ARG3', 'I-ARG3', 'I-ARG3', 'I-ARG3', 'O']}],
            'words': ['Twitter', 'confirms', 'sale', 'of', 'company', 'to', 'Elon', 'Musk', 'for', '$', '44',
                      'billion', '.']
        }
    ]

    def setUp(self):
        self.predictor = WebNomSRLPredictor()

    def test_predict(self):
        srl_pred = self.predictor.predict(self.sentences[0])
        self.assertIsInstance(srl_pred, NomSRLResult)
        self.assertDictEqual(asdict(srl_pred), self.references[0])

    def test_batch_predict(self):
        predictions = self.predictor.batch_predict(self.sentences)
        for srl_pred, srl_gold in zip(predictions, self.references):
            self.assertIsInstance(srl_pred, NomSRLResult)
            self.assertDictEqual(asdict(srl_pred), srl_gold)


if __name__ == '__main__':
    unittest.main()
