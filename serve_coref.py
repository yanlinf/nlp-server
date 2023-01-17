import argparse
import traceback
from allennlp.predictors.predictor import Predictor
from aiohttp import web

routes = web.RouteTableDef()


@routes.post('/coref')
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
    parser.add_argument('-p', '--port', default=8985)
    args = parser.parse_args()

    url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz'
    model = Predictor.from_path(url)

    app = web.Application()
    app['model'] = model
    app.add_routes(routes)
    web.run_app(app, port=args.port)


if __name__ == '__main__':
    main()
