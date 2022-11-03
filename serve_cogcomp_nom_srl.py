import argparse
from aiohttp import web
from typing import List
import traceback
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
        nom_srl_res = self.nom_srl_predictor.predict(
            sentence=sentence,
            indices=[i for i, x in enumerate(nom_id_res['nominals']) if x == 1]
        )
        assert isinstance(nom_srl_res, dict)
        return nom_srl_res

    def predict_batch_json(self, inputs: List[dict]) -> List[dict]:
        nom_id_res = self.nom_id_predictor.predict_batch_json(inputs)
        assert len(nom_id_res) == len(inputs)

        # The original implementation at https://github.com/CogComp/SRL-English
        # with `convert_id_to_srl_input` seems to be buggy
        nom_srl_inputs = [{
            'sentence': dic['sentence'],
            'indices': [i for i, x in enumerate(res['nominals']) if x == 1],
        } for res, dic in zip(nom_id_res, inputs)]
        assert len(nom_srl_inputs) == len(inputs)

        nom_srl_res = self.nom_srl_predictor.predict_batch_json(nom_srl_inputs)
        assert len(nom_srl_res) == len(inputs)
        assert all(isinstance(d, dict) for d in nom_srl_res)

        return nom_srl_res

    def _convert_id_to_srl_input(self, nom_id_res: dict) -> dict:
        """
        deprecated, adapted from https://github.com/CogComp/SRL-English/blob/main/convert_id_to_srl_input.py
        """
        indices = [idx for idx in range(len(nom_id_res['nominals'])) if nom_id_res['nominals'][idx] == 1]
        words, indices = self._shift_indices_for_empty_strings(nom_id_res['words'], indices)
        return {
            'sentence': ' '.join(words),
            'indices': indices
        }

    def _shift_indices_for_empty_strings(self, words, indices):
        """
        deprecated, adapted from https://github.com/CogComp/SRL-English/blob/main/convert_id_to_srl_input.py
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


def empty_nom_frame():
    return {
        'nominals': [],
        'words': [],
    }


@routes.post('/cogcomp_nom_srl')
async def handle_srl(request):
    model = request.app['model']
    params = await request.json()
    try:
        if isinstance(params, list):
            inputs = [d for d in params if d['sentence'].strip() != '']
            predictions = model.predict_batch_json(inputs)
            res = []
            for d in params:
                if d['sentence'].strip() == '':
                    res.append(empty_nom_frame())
                else:
                    res.append(predictions.pop(0))
            assert len(res) == len(params)
        else:
            if params['sentence'].strip() == '':
                res = empty_frame()
            else:
                res = model.predict(**params)
    except Exception as e:
        print(traceback.format_exc())
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
