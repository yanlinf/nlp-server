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

Get dependency parse:

```bash
import requests
url = 'http://127.0.0.1:8982/parse'
requests.post(url, json=[{'sentence': 'I love you'}, {'sentence': 'asdf'}]).json()
# [{'words': ['I', 'love', 'you'],
#  'pos': ['PRON', 'VERB', 'PRON'],
#  'predicted_dependencies': ['nsubj', 'root', 'dep'],
#  ...
```

Get verb sense SRL predictions:

```bash
import requests
url = 'http://127.0.0.1:8983/cogcomp_verb_srl'
requests.post(url, json=[{'sentence': 'I love you'}]).json()
# [{'verbs': [{'verb': 'love',
#     'sense': '1.0',
#     'description': '[ARG0: I] [V: love] [ARG1: you]',
#     'tags': ['B-ARG0', 'B-V', 'B-ARG1']}],
#   'words': ['I', 'love', 'you']}]
```

Get nominal sense SRL predictions:

```bash
import requests
url = 'http://127.0.0.1:8984/cogcomp_nom_srl'
requests.post(url, json=[{'sentence': 'Twitter confirms sale of company to Elon Musk for $44 billion.'}]).json()
# [{'nominals': [{'nominal': 'sale',
#     'sense': '01',
#     'predicate_index': [2],
#     'description': '[ARG0: Twitter] [Support: confirms] sale [ARG1: of company] [ARG2: to Elon Musk] [ARG3: for $ 44 billion]',
#     'tags': ['B-ARG0','B-Support','O','B-ARG1','I-ARG1','B-ARG2','I-ARG2','I-ARG2','B-ARG3','I-ARG3','I-ARG3','I-ARG3']}],
#   'words': ['Twitter','confirms','sale','of','company','to','Elon','Musk','for','$','44','billion']}]
```