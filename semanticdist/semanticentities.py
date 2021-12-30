import requests

def get_entities_for_text(text, context):
    _params = {
        'text': text,
        'lang': 'AGNOSTIC',
        'annType': 'NAMED_ENTITIES',
        'key': context['babelfy_key']
    }
    return requests.post("https://babelfy.io/v1/disambiguate", data=_params).json()
