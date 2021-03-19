from abc import ABC
from typing import List, Tuple

import spacy
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab

from config import global_output_directory_name
from utils import PandasUtil, FileUtil


class FileManipulator(ABC):
    def __init__(self):
        english_model = spacy.load("en_core_web_sm")

    def get_output_directory(self, subdirectory):
        return f'{global_output_directory_name}/{subdirectory}'

    def remove_unserializable_results(self, doc):
        doc.user_data = {}
        for x in dir(doc._):
            if x in ['get', 'set', 'has']: continue
            setattr(doc._, x, None)
        for token in doc:
            for x in dir(token._):
                if x in ['get', 'set', 'has']: continue
                setattr(token._, x, None)
        return doc

    def save(self, doc: Doc, objects: List, file_name: str, file_subdirectory: str):
        output_directory = self.get_output_directory(file_subdirectory)
        FileUtil.create_directory_if_not_exists(output_directory)
        doc, objects_column_names = self.prepare_doc_for_saving(doc, objects)
        doc.to_disk(f'{output_directory}/{file_name}')
        PandasUtil.write_object_list_as_dataframe_file(doc._.entities, file_name, f'{output_directory}/as_df', columns=objects_column_names)

    def read_and_parse(self, file_name: str, file_subdirectory: str, doc: Doc = None) -> List:
        output_directory = self.get_output_directory(file_subdirectory)
        read_doc = Doc(Vocab()).from_disk(f'{output_directory}\{file_name}')
        return self.parse_read_doc(read_doc)

    def prepare_doc_for_saving(self, doc: Doc, objects: List, extractor_name: str) -> Tuple[Doc, List]:
        raise NotImplementedError()

    def parse_read_doc(self, doc: Doc = None) -> List:
        raise NotImplementedError()


class EntityExtractorFileManipulator(FileManipulator):
    def parse_read_doc(self, doc: Doc = None) -> List:
        return [doc.char_span(span[0], span[1]) for span in doc._.entities]

    def prepare_doc_for_saving(self, doc: Doc, objects: List, extractor_name: str) -> Tuple[Doc, List]:
        entities_columns = ['start_char', 'end_char', 'entity_type', 'entity_id', 'text', 'sentence', 'sentence_index']
        if extractor_name.find('foodner') > -1:
            entities_columns += ['foodon', 'hansard', 'hansardClosest', 'hansardParent', 'snomedct', 'synonyms']
            entities = [
            (span.start_char, span.end_char, span._.entity_type, span._.entity_id, span.text, span.sent.text.strip(), list(doc.sents).index(span.sent),
             span._.foodon, span._.hansard, span._.hansardClosest, span._.hansardParent, span._.snomedct, span._.synonyms
             )
            if span.sent in doc.sents else None for span in objects]

        else:
            entities = [(span.start_char, span.end_char, span._.entity_type,span._.entity_id, span.text, span.sent.text.strip(),
                         list(doc.sents).index(span.sent)) if span.sent in doc.sents else None for span in objects]
        #doc = self.remove_unserializable_results(doc)
        doc._.entities = entities

        return doc, entities_columns


class RelationExtractorFileManipulator(FileManipulator):
    def parse_read_doc(self, doc: Doc = None) -> List:
        return doc._.relations

    def prepare_doc_for_saving(self, doc: Doc, objects: List, extractor_name: str) -> Tuple[Doc, List]:
        doc = self.remove_unserializable_results(doc)
        print(objects)
        relations = [(triple[0], triple[1], triple[2]) for triple in objects]
        doc._.relations = relations
        relations_columns = ['subject', 'relation', 'object']
        return doc, relations_columns
