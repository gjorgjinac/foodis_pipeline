import os
from typing import List
import pandas as pd
from pyvis.network import Network
from spacy.lang.en import English
import matplotlib.pyplot as plt
import networkx as nx
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from pandas.errors import EmptyDataError

from config import global_output_directory_name


class PandasUtil:
    @staticmethod
    def write_object_list_as_dataframe_file(object_list: List, file_name: str, file_directory: str = '.',
                                            df_separator: str = ',', columns: List[str] = None):
        #dictionary_list = [object.__dict__ for object in object_list]
        FileUtil.create_directory_if_not_exists(file_directory)
        df_to_write = pd.DataFrame(object_list)
        if columns is not None:
            df_to_write = pd.DataFrame(object_list, columns=columns)
        df_to_write.to_csv(f'{file_directory}/{file_name}', sep=df_separator)

    @staticmethod
    def read_dataframe_file_as_object_list(file_name: str, file_directory: str = '.', df_separator: str = ',', nan_replacement = None):
        read_df = PandasUtil.read_dataframe_file(file_name, file_directory, df_separator, nan_replacement)
        return read_df.to_dict(orient='records')

    @staticmethod
    def read_dataframe_file(file_name: str, file_directory: str = '.', df_separator: str = ',', nan_replacement = None):
        try:
            read_df = pd.read_csv(os.path.join(file_directory, file_name), sep = df_separator, index_col=[0])
            if nan_replacement is not None:
                read_df = read_df.fillna(nan_replacement)
            return read_df
        except EmptyDataError:
            print('empty data error')
            return pd.DataFrame()
class PrintUtil:
    @staticmethod
    def print_objects_with_labels(objects_with_labels: List):
        for object_with_label in objects_with_labels:
            print(f'{object_with_label[0]}:\n{object_with_label[1]}')
        print()


class TextProcessingUtil:
    @staticmethod
    def split_into_sentences(text: str) -> List[Span]:
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        document = nlp(text)
        return list(filter(lambda s: not TextProcessingUtil.is_empty(s.text.strip()), document.sents))

    @staticmethod
    def is_empty(text) -> bool:
        return len(text.strip()) == 0

    @staticmethod
    def get_text_between_words(sentence, word1, word2) -> str:
        word1_index = sentence.find(word1)
        word2_index = sentence.find(word2)
        if word1_index < word2_index:
            return sentence[word1_index + len(word1): word2_index]
        else:
            return sentence[word2_index + len(word2): word1_index]

    @staticmethod
    def remove_text_between_brackets(text) -> str:
        while True:
            start = text.find('(')
            end = text.find(')')
            if start != -1 and end != -1:
                text = text[start + 1:end]
            else:
                break
        return text

class FileUtil:
    @staticmethod
    def read_file(file_name: str, file_directory: str = '.') -> str:
        file_path = f'{file_directory}/{file_name}'
        file = open(file_path, mode='r', encoding='utf-8' )
        file_content = file.read()
        file.close()
        return file_content

    @staticmethod
    def create_directory_if_not_exists(directory:str):
        if not os.path.isdir(directory):
            os.makedirs(os.path.join(os.getcwd(), directory))

class GraphUtil:
    @staticmethod
    def generate_graph_from_triples(triples) -> nx.Graph:
        G = nx.DiGraph()
        for triple in triples:
            G.add_nodes_from([
                (triple[0], {"type": "food"}),
                (triple[2], {"type": "disease"}),
            ])
            G.add_edge(triple[0], triple[2], relationship = triple[1])
        return G

    @staticmethod
    def draw_graph(G, food_node_color = 'palegreen', relation_node_color='plum', disease_node_color='lightcoral'):
        node_type_color_map = {'food': food_node_color, 'relation': relation_node_color, 'disease': disease_node_color}
        pos = nx.spring_layout(G)
        plt.figure()
        node_colors = [node_type_color_map[node_type[1]] for node_type in G.nodes.data('type')]
        nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
                node_size=500,
                node_color=node_colors,
                labels={node: node for node in G.nodes()})
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'relationship'))
        plt.axis('off')
        plt.show()

    @staticmethod
    def draw_graph_with_pyvis(triples_df, dataset_name,cause_column, treat_column,food_column='term1', disease_column='term2',
                              food_node_color = 'green', disease_node_color='darkgrey', cause_color='lightcoral', treat_color='palegreen'):
        nt = Network("800px", "100%")
        nt.heading=f'Food-disease relations identified in dataset: {dataset_name}'
        triples_df[food_column]=triples_df[food_column].apply(lambda x: x.lower())
        triples_df[disease_column] = triples_df[disease_column].apply(lambda x: x.lower())
        unique_pairs = triples_df.groupby([food_column, disease_column, 'entity_id_y','foodon','snomedct','hansard','hansardClosest','hansardParent', 'synonyms']).mean()[[cause_column, treat_column]]
        unique_pairs = unique_pairs[(unique_pairs[cause_column] > 0) | (unique_pairs[treat_column] > 0) ]
        print(unique_pairs)

        results=[]
        for index_columns in unique_pairs.index:
            term1, term2, doid, foodon, snomedct, hansard, hansard_closest, hansard_parent, synonyms = index_columns
            all_evidence = triples_df[triples_df[food_column] == term1][triples_df[disease_column] == term2][
                'relation_candidates']
            if len(set(all_evidence.values)) > 1:

                all_evidence = '______'.join(set(all_evidence.values))
                label_term1 = triples_df[triples_df[food_column]==term1]['term1'].value_counts().idxmax()
                label_term2 = triples_df[triples_df[disease_column] == term2]['term2'].value_counts().idxmax()
                nt.add_node(term1, label=label_term1, color=food_node_color, size=15)
                nt.add_node(term2, label=label_term2, color=disease_node_color, size=15)
                row = unique_pairs.loc[index_columns, :]
                edge_color = cause_color if row[cause_column] > row[treat_column] else treat_color if row[cause_column] < row[treat_column] else 'purple'
                results.append({'term1': label_term1, 'term2': label_term2, 'evidence': all_evidence, 'treat': row[treat_column], 'cause': row[cause_column],
                                   'doid': doid, 'foodon': foodon, 'snomedct': snomedct, 'hansard': hansard, 'hansardClosest': hansard_closest, 'hansardParent': hansard_parent, 'synonyms': synonyms})
                edge_label = f'treat: {row[treat_column]} cause: {row[cause_column]} \n{all_evidence}'
                nt.add_edge(term1, term2, title=edge_label, color=edge_color, width=5)
        nt.show_buttons(filter_=['physics','nodes','edges','selection','layout'])
        pd.DataFrame(results).to_csv('visualization_triples_sum.csv')
        print(len(nt.edges))
        # nt.enable_physics(True)
        nt.show("nx.html")


def get_dataset_output_dir(dataset):
    return os.path.join(global_output_directory_name, dataset)

def write_to_dataset_dir(df, file_name, dataset):
    dataset_output_dir = get_dataset_output_dir(dataset)
    df.to_csv(os.path.join(dataset_output_dir, file_name))

def read_from_dataset_dir(file_name, dataset):
    dataset_output_dir = get_dataset_output_dir(dataset)
    return pd.read_csv(os.path.join(dataset_output_dir, file_name), index_col=[0])


def add_span_extensions():
    Doc.set_extension("relations", default=None)
    Doc.set_extension("entities", default=None)
    for span_extension in ['entity_type', 'entity_id', 'foodon', 'hansard', 'hansardClosest', 'hansardParent',
                           'snomedct', 'synonyms']:
        Span.set_extension(span_extension, default=None)