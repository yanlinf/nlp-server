import argparse
import traceback
from allennlp.predictors.predictor import Predictor
from aiohttp import web

routes = web.RouteTableDef()


@routes.post('/srl')
async def handle_srl(request):
    model = request.app['model']
    params = await request.json()
    try:
        if isinstance(params, list):
            res = model.predict_batch_json(params)
        else:
            res = model.predict(**params)
    except Exception as e:
        print(traceback.format_exc())
        return web.json_response({'error': 'Invalid request'})
    return web.json_response(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=8981)
    args = parser.parse_args()

    url = 'https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz'
    model = Predictor.from_path(url, cuda_device=0)

    app = web.Application()
    app['model'] = model
    app.add_routes(routes)
    web.run_app(app, port=args.port)


if __name__ == '__main__':
    main()
