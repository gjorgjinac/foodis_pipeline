import string

import spacy
import networkx as nx
from nltk import word_tokenize


class ShortestDependencyPathRelation:
    name: str

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.name = 'sdp'

    def check_if_token_in_entities(self, token, entity1, words_in_e1, entity2, words_in_e2, nodes):
        node = token.lower_
        if token.lower_ in words_in_e1:
            node = entity1
        if token.lower_ in words_in_e2:
            node = entity2
        if token.lower_ in words_in_e1 and token.lower_ in words_in_e2:
            if entity1 in nodes:
                node = entity2
            else:
                node = entity1
        return node

    def get_sdp_for_sentence(self, sentence, entity1, entity2):
        entity1_index = sentence.find(entity1)
        entity2_index = sentence.find(entity2)
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).replace('  ', ' ')
        doc = self.nlp(sentence)
        print(f'sentence:{doc}')
        # print(f'e1: {entity1}   e2: {entity2}')
        # Load spacy's dependency tree into a networkx graph
        entity1, entity2 = entity1.lower(), entity2.lower()
        words_in_e1 = word_tokenize(entity1)
        words_in_e2 = word_tokenize(entity2)
        edges = []
        nodes = []
        for token in doc:
            for child in token.children:
                node1 = self.check_if_token_in_entities(token, entity1, words_in_e1, entity2, words_in_e2, nodes)
                node2 = self.check_if_token_in_entities(child, entity1, words_in_e1, entity2, words_in_e2, nodes)
                edges.append(('{0}'.format(node1), '{0}'.format(node2)))
                [nodes.append(n) for n in [node1, node2]]
        graph = nx.Graph(edges)
        # Get the length and path
        try:
            source, target = (entity1, entity2) if entity2_index > entity1_index else (entity2, entity1)
            return [nx.shortest_path(graph, source=source, target=target)]
        except Exception:
            return ' '

    def extract_relation(self, df):
        df['relation_candidates'] = df.apply(lambda sentence: self.get_sdp_for_sentence(sentence, 'XXX', 'YYY'))
        return df


class WordContextWindowRelation:
    context_window: int
    name: str

    def __init__(self, context_window):
        self.context_window = context_window
        self.name = f'w{context_window}'

    def extract_relation(self, df):
        df['relation_candidates'] = df['relation_candidates'].apply(
            lambda sentence: self.get_words_in_context_window(sentence))
        return df

    def get_words_in_context_window(self, sentence):
        masked_sentence = sentence.replace('XXX', ' XXX ').replace('YYY', ' YYY ').replace('  ', ' ')
        words_in_sentence = word_tokenize(masked_sentence)

        try:
            x_index = words_in_sentence.index('XXX')
            y_index = words_in_sentence.index('YYY')
            start, end = (x_index, y_index) if x_index <= y_index else (y_index, x_index)
            start = start - self.context_window
            start = 0 if start < 0 else start
            end = end + self.context_window + 1
            end = len(words_in_sentence) if end > len(words_in_sentence) else end
            return ' '.join(words_in_sentence[start: end])
        except ValueError:
            print(sentence)
            print(f'ValueError: {words_in_sentence}')
            return ' '
