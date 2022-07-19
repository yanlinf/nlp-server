# nlp-server

## Usage

Run fasttext server:

```bash
nohup python -u serve_fasttext.py --model /path/to/wiki.en.bin --port 8980 &
```

Query word vectors (python):

```python
import requests, json

url = 'http://127.0.0.1:8980/fasttext'
tokens = ['apple', 'asdf@andrew.cmu.edu']
resp = requests.get(url, params={'tokens': json.dumps(tokens)}).json()
vectors = resp['vectors']
assert vectors.shape == (2, 300)
```

Query word vectors (shell):

```bash
curl http://127.0.0.1:8980/fasttext?tokens=[%22apple%22]
```

Run [SRL server](https://demo.allennlp.org/semantic-role-labeling):

```bash
nohup python -u serve_srl.py --port 8981 &
```

Get SRL predictions:

```bash
import requests
url = 'http://127.0.0.1:8981/srl'
requests.post(url, json=[{'sentence': 'I love you'}, {'sentence': 'asdf'}]).json()
# [{'verbs': [{'verb': 'love',
#    'description': '[ARG0: I] [V: love] [ARG1: you]',
#    'tags': ['B-ARG0', 'B-V', 'B-ARG1']}],
#  'words': ['I', 'love', 'you']},
# {'verbs': [], 'words': ['asdf']}]
```

Run [dependency parser](https://demo.allennlp.org/dependency-parsing):

```bash
nohup python -u serve_parser.py --port 8982 &
```

Get SRL predictions:

```bash
import requests
url = 'http://127.0.0.1:8982/parse'
requests.post(url, json=[{'sentence': 'I love you'}, {'sentence': 'asdf'}]).json()
# [{'words': ['I', 'love', 'you'],
#  'pos': ['PRON', 'VERB', 'PRON'],
#  'predicted_dependencies': ['nsubj', 'root', 'dep'],
#  ...
```
