import argparse
import traceback
from aiohttp import web
from cogcomp_srl.verb_sense_srl import SenseSRLPredictor

routes = web.RouteTableDef()


def empty_verb_frame():
    return {
        'verbs': [],
        'words': [],
    }


@routes.post('/cogcomp_verb_srl')
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
                    res.append(empty_verb_frame())
                else:
                    res.append(predictions.pop(0))
            assert len(res) == len(params)
        else:
            if params['sentence'].strip() == '':
                res = empty_verb_frame()
            else:
                res = model.predict(**params)
    except Exception as e:
        print(traceback.format_exc())
        return web.json_response({'error': 'Invalid request'})
    return web.json_response(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=8983)
    args = parser.parse_args()

    path = 'checkpoints/cogcomp-verb-sense-srl.tar.gz'
    model = SenseSRLPredictor.from_path(
        path,
        predictor_name='sense-semantic-role-labeling',
        cuda_device=0
    )

    app = web.Application()
    app['model'] = model
    app.add_routes(routes)
    web.run_app(app, port=args.port)


if __name__ == '__main__':
    main()
