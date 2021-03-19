import sys
import traceback
from typing import List, Tuple, Any

import spacy
import stanza
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from extractors.extractor_base_classes import EntityExtractor

from saber_local.saber import Saber

class SaberBioMedExtractor(EntityExtractor):
    english_model: Language
    model: any
    def __init__(self, model_name='diso', save_extractions=True):
        super().__init__('saber_{model_name}'.format(model_name=model_name), save_extractions)
        self.model = Saber()
        self.model.load(model_name.upper())
        self.english_model = spacy.load('en_core_web_sm')

    def extract_entity(self, doc: Doc) -> List[Span]:
        return self.extract_entities_with_saber_model(doc)

    def extract_entities_with_saber_model(self, spacy_doc):
        try:
            processed_doc = self.model.annotate(spacy_doc.text, ground=True, coref=False)
        except:
            print('Error happened')
            traceback.print_exc(file=sys.stdout)
            return []

        entities = processed_doc['ents']
        spacy_doc = self.english_model(processed_doc['text'])
        entities_as_spacy_spans = []

        for e in entities:
            entity_span = spacy_doc.char_span(e['start'], e['end'])
            if entity_span is None:
                print('cannot find: {e}'.format(e = e["text"]))
                continue
            entity_span._.entity_type = e['label']
            if 'xrefs' in e.keys():
                entity_span._.entity_id = ';'.join([ref['id'] for ref in e['xrefs']])
            entities_as_spacy_spans.append(entity_span)
        return entities_as_spacy_spans
