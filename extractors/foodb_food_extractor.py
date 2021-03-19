
from typing import List

import pandas as pd
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from extractors.extractor_base_classes import EntityExtractor


class FoodbFoodExtractor(EntityExtractor):
    spacy_model: Language

    def __init__(self, save_extractions=True, include_scientific=True, include_non_scientific = False):
        super().__init__(f'foodb_scientific_{include_scientific}_non_scientific_{include_non_scientific}', save_extractions)
        food_df = pd.read_csv('foodb/Food.csv').fillna('')
        self.food_names = set()
        if include_non_scientific:
            self.food_names=self.food_names.union(set(food_df['name']))
        if include_scientific:
            self.food_names = self.food_names.union(set(food_df['name_scientific']))
        self.food_names = list(filter(lambda x: len(x) > 0, self.food_names))
        self.food_names = [f.lower() for f in self.food_names]


    def extract_entity(self, doc: Doc) -> List[Span]:
        food_spans = []
        for food in self.food_names:
            food_index = doc.text.lower().find(food)
            if food_index > -1:
                food_spans.append(doc.char_span(food_index, food_index + len(food)))
        return food_spans



