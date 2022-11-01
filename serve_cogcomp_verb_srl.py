import argparse
from aiohttp import web
import sys
sys.path.append('cogcomp_srl')
from cogcomp_srl.verb_sense_srl import SenseSRLPredictor

routes = web.RouteTableDef()


@routes.post('/cogcomp_verb_srl')
async def handle_srl(request):
    model = request.app['model']
    params = await request.json()
    try:
        if isinstance(params, list):
            res = model.predict_batch_json(params)
        else:
            res = model.predict(**params)
    except Exception as e:
        print(e)
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
