import json
from typing import List

import requests
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from extractors.extractor_base_classes import EntityExtractor


def send_request_to_foodviz(text):
    cookies = {
        '_ga': 'GA1.2.1090152998.1599672134',
    }

    headers = {
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Accept': 'application/json, text/plain, */*',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
        'Referer': 'http://foodviz.env4health.finki.ukim.mk/',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    params = (
        ('text', text),
        ('model', 'bert-model-food-classification-e94-0.0005.bin'),
    )

    response = requests.get('http://foodviz.env4health.finki.ukim.mk/predict', headers=headers, params=params, cookies=cookies, verify=False)
    return json.loads(response.text)['tokens']

def extract_food_entities_from_text(text):
    tokens = send_request_to_foodviz(text)
    tokens = list(filter(lambda x: x['tag']!='O', tokens))
    return tokens

class FoodnerWebFoodExtractor(EntityExtractor):


    def __init__(self, save_extractions=False):
        super().__init__('foodner_web', save_extractions=save_extractions)


    def extract_entity(self, doc: Doc) -> List[Span]:
        food_entities = self.find_food_entities(doc)
        return self.__convert_food_words_to_doc(doc, food_entities)

    def find_food_entities(self, doc: Doc) -> List:
        processed_words = extract_food_entities_from_text(doc.text)
        food_words = []
        for processed_word in processed_words:
            if processed_word['tag']!='O' or processed_word['otherTags'] is not None:
                food_words.append(processed_word)
        return food_words

    def __string_value_from_dict_or_none(self, dict, dict_key):
        if dict_key in dict.keys():
            return dict[dict_key] if type(dict[dict_key]) is not list else '***'.join(dict[dict_key])
        return None

    def __add_tags_to_span(self, food_span, processed_word):
        if processed_word['otherTags'] is not None:
            tags = processed_word['otherTags']
            food_span._.foodon = self.__string_value_from_dict_or_none(tags, 'foodon')
            food_span._.hansard = self.__string_value_from_dict_or_none(tags, 'hansard')
            food_span._.hansardClosest = self.__string_value_from_dict_or_none(tags, 'hansardClosest')
            food_span._.hansardParent = self.__string_value_from_dict_or_none(tags, 'hansardParent')
            food_span._.snomedct = self.__string_value_from_dict_or_none(tags, 'snomedct')
            food_span._.synonyms = self.__string_value_from_dict_or_none(tags, 'synonyms')
        return food_span
    def __convert_food_words_to_doc(self, doc, food_words):
        food_spans = []
        for processed_word in food_words:
            food_span = doc[processed_word['start']: processed_word['start'] + processed_word['numTokens']]
            food_span=self.__add_tags_to_span(food_span, processed_word)
            food_spans.append(food_span)
        return food_spans


