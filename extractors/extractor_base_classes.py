from abc import ABC

from pandas import DataFrame
from spacy.language import Language


class Extractor(ABC):
    english_model: Language
    save_extractions: bool

    def __init__(self, save_extractions=True):
        self.english_model = spacy.load('en_core_web_sm')
        self.save_extractions = save_extractions

    def extract_from_text(self, text, *args):
        doc = self.english_model(text)
        return self.extract(doc, *args)

    def extract_from_file(self, file_name, file_directory, dataset='', *args):
        text = FileUtil.read_file(file_name, file_directory)
        # print(text)
        doc = self.english_model(text)
        return self.extract(doc, file_name, dataset, *args)

    def extract(self, *args):
        raise NotImplementedError()


from typing import List, Tuple, Union

import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from extractors.file_manipulators import RelationExtractorFileManipulator, FileManipulator, \
    EntityExtractorFileManipulator
from utils import FileUtil, PandasUtil
import pandas as pd

class EntityExtractor(Extractor):
    name: str
    file_manipulator: FileManipulator

    def __init__(self, name, save_extractions=True):
        super(EntityExtractor, self).__init__(save_extractions)
        self.name = name
        self.file_manipulator = EntityExtractorFileManipulator()

    def extract_entity(self, doc: Doc) -> List[Span]:
        raise NotImplementedError()

    def extract(self, doc: Doc, file_name=None, dataset_source='', save_entities=True) -> Union[
        list, DataFrame, List[Span]]:
        output_directory = f'{dataset_source}/{self.name}'
        try:
            return self.file_manipulator.read_and_parse(file_name, output_directory, doc)
        except Exception:
            #print('recalculating')
            spans = self.extract_entity(doc)
            spans = list(filter(lambda s: s is not None, spans))
            doc, objects_column_names = self.file_manipulator.prepare_doc_for_saving(doc, spans, self.name)
            return pd.DataFrame(doc._.entities, columns=objects_column_names)




class RelationExtractor(Extractor):
    name: str
    file_manipulator: FileManipulator

    def __init__(self, name, save_extractions=True):
        super().__init__(save_extractions)
        self.name = name
        self.file_manipulator = RelationExtractorFileManipulator()

    def extract_relation(self, doc: Doc, first_argument_candidates, second_argument_candidates, file_name,
                         file_directory) -> List[Tuple]:
        raise NotImplementedError()

    def extract(self, doc: Doc, file_name=None, dataset=None, first_argument_candidates=None,
                second_argument_candidates=None,
                foodis_model_name=None) -> List[Tuple]:
        final_relations_directory_name = f'{dataset}/{self.name}/{foodis_model_name}'
        all_relations_directory_name = f'{dataset}/{self.name}'

        try:
            return self.file_manipulator.read_and_parse(file_name, final_relations_directory_name, doc)
        except Exception:
            print('recalculating')
            relations = self.extract_relation(doc, first_argument_candidates, second_argument_candidates, file_name,
                                              all_relations_directory_name)
            if self.save_extractions and len(relations) > 0:
                self.file_manipulator.save(doc, relations, file_name, final_relations_directory_name)
            return relations
