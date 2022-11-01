import argparse
from aiohttp import web
from typing import List
import sys

sys.path.append('cogcomp_srl')
from cogcomp_srl.id_nominal import NominalIdPredictor
from cogcomp_srl.nominal_sense_srl import NomSenseSRLPredictor

routes = web.RouteTableDef()


class NomSRLPredictor:
    """
    A predictor that integrates "nombank-id" and "nombank-sense-srl"
    """

    def __init__(self, nom_id_predictor: NominalIdPredictor,
                 nom_srl_predictor: NomSenseSRLPredictor):
        self.nom_id_predictor = nom_id_predictor
        self.nom_srl_predictor = nom_srl_predictor

    @classmethod
    def from_path(cls, nom_id_model_path: str, nom_sense_srl_model_path: str, cuda_device: int = -1):
        nom_id_predictor = NominalIdPredictor.from_path(
            nom_id_model_path,
            predictor_name='nombank-id',
            cuda_device=cuda_device
        )
        nom_srl_predictor = NomSenseSRLPredictor.from_path(
            nom_sense_srl_model_path,
            predictor_name='nombank-sense-srl',
            cuda_device=cuda_device
        )
        return cls(nom_id_predictor, nom_srl_predictor)

    def predict(self, sentence: str) -> dict:
        nom_id_res = self.nom_id_predictor.predict(sentence)
        nom_srl_inputs = self._convert_id_to_srl_input(nom_id_res)
        nom_srl_res = self.nom_srl_predictor.predict(
            sentence=nom_srl_inputs['sentence'],
            indices=nom_srl_inputs['indices']
        )
        assert isinstance(nom_srl_res, dict)
        return nom_srl_res

    def predict_batch_json(self, inputs: List[dict]) -> List[dict]:
        nom_id_res = self.nom_id_predictor.predict_batch_json(inputs)
        assert len(nom_id_res) == len(inputs)

        nom_srl_inputs = [self._convert_id_to_srl_input(dic) for dic in nom_id_res]
        assert len(nom_srl_inputs) == len(inputs)

        nom_srl_res = self.nom_srl_predictor.predict_batch_json(nom_srl_inputs)
        assert len(nom_srl_res) == len(inputs)
        assert all(isinstance(d, dict) for d in nom_srl_res)

        return nom_srl_res

    def _convert_id_to_srl_input(self, nom_id_res: dict) -> dict:
        """
        adapted from https://github.com/CogComp/SRL-English/blob/main/convert_id_to_srl_input.py
        """
        indices = [idx for idx in range(len(nom_id_res['nominals'])) if nom_id_res['nominals'][idx] == 1]
        words, indices = self._shift_indices_for_empty_strings(nom_id_res['words'], indices)
        return {
            'sentence': ' '.join(words),
            'indices': indices
        }

    def _shift_indices_for_empty_strings(self, words, indices):
        """
        adapted from https://github.com/CogComp/SRL-English/blob/main/convert_id_to_srl_input.py
        """
        shiftleft = 0
        new_indices = []
        new_words = []
        for idx, word in enumerate(words):
            if word == '' or word.isspace():
                shiftleft += 1
            else:
                if idx in indices:
                    new_indices.append(idx - shiftleft)
                new_words.append(word)
        return new_words, new_indices


@routes.post('/cogcomp_nom_srl')
async def handle_srl(request):
    model = request.app['model']
    params = await request.json()
    try:
        if isinstance(params, list):
            res = model.predict_batch_json(params)
        else:
            res = model.predict(**params)
    except Exception as e:
        return web.json_response({'error': 'Invalid request'})
    return web.json_response(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=8984)
    args = parser.parse_args()

    nom_id_model_path = 'checkpoints/cogcomp-nom-id.tar.gz'
    nom_sense_srl_model_path = 'checkpoints/cogcomp-nom-sense-srl.tar.gz'
    model = NomSRLPredictor.from_path(
        nom_id_model_path,
        nom_sense_srl_model_path,
        cuda_device=0
    )

    app = web.Application()
    app['model'] = model
    app.add_routes(routes)
    web.run_app(app, port=args.port)


if __name__ == '__main__':
    main()
