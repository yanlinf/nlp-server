import argparse
import asyncio
import json
from fasttext import load_model
from aiohttp import web
import numpy as np


class WordToVectorDict:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, word):
        # Check if mean for word split needs to be done here
        return np.mean([self.model.get_word_vector(w) for w in word.split(" ")], axis=0)


routes = web.RouteTableDef()


@routes.get('/fasttext')
async def handle_fasttext(request):
    stov = request.app['stov']
    tokens = request.query.get('tokens')
    if tokens is None:
        return web.json_response({'error': 'parameter `tokens` not found'})

    tokens = json.loads(tokens)
    return web.json_response({
        'vectors': [stov[w].tolist() for w in tokens]
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-p', '--port', default=8980)
    args = parser.parse_args()

    print(f"Loading fasttext model now from {args.model}")
    model = load_model(args.model)
    stov = WordToVectorDict(model)

    app = web.Application()
    app['stov'] = stov
    app.add_routes(routes)
    web.run_app(app, port=args.port)


if __name__ == '__main__':
    main()
