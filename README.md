# nlp-server

## Usage

### Start server

```bash
nohup python -u serve_fasttext.py --model /path/to/wiki.en.bin 
                                  --port 8980
```

### Query word vectors

#### Python

```python
import requests, json

url = 'https://127.0.0.1:8980/fasttext'
tokens = ['apple', 'asdf@andrew.cmu.edu']
resp = requests.get(url, params={'tokens', json.dumps(tokens)}).json()
vectors = resp['vectors']
assert vectors.shape == (2, 300)
```

#### Shell

```bash
curl http://127.0.0.1:8980/fasttext?tokens=[%22apple%22]
```
