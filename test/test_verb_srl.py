import unittest
from dataclasses import asdict
from clients import WebVerbSRLPredictor, VerbSRLResult


class VerbSRLTestcase(unittest.TestCase):
    sentences = ['Twitter confirms sale of company to Elon Musk for $44 billion.']
    references = [
        {
            'verbs': [{'verb': 'confirms', 'sense': '1.0',
                       'description': '[ARG0: Twitter] [V: confirms] [ARG1: sale of company to Elon Musk for $ 44 billion] .',
                       'tags': ['B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1',
                                'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']}],
            'words': ['Twitter', 'confirms', 'sale', 'of', 'company', 'to', 'Elon', 'Musk', 'for', '$', '44',
                      'billion', '.']
        }
    ]

    def setUp(self):
        self.predictor = WebVerbSRLPredictor()

    def test_predict(self):
        srl_pred = self.predictor.predict(self.sentences[0])
        self.assertIsInstance(srl_pred, VerbSRLResult)
        self.assertDictEqual(asdict(srl_pred), self.references[0])

    def test_batch_predict(self):
        predictions = self.predictor.batch_predict(self.sentences)
        for srl_pred, srl_gold in zip(predictions, self.references):
            self.assertIsInstance(srl_pred, VerbSRLResult)
            self.assertDictEqual(asdict(srl_pred), srl_gold)


if __name__ == '__main__':
    unittest.main()
