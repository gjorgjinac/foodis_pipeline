import pandas as pd

class CauseTreatProcessor():
    def extract_relation(self, row):

        sentence = row['sentence']
        if len(row['term1']) > len(row['term2']):
            sentence = sentence.replace(row['term1'], 'XXX').replace(row['term2'], 'YYY')
        else:
            sentence = sentence.replace(row['term2'], 'YYY').replace(row['term1'], 'XXX')
        if sentence.find('XXX') == -1 or sentence.find('YYY') == -1:
            sentence = sentence.replace('XXX', 'XXXYYY').replace('YYY', 'XXXYYY')
        return sentence
        term_1_start = sentence.find(row['term1'])
        term_1_end = term_1_start + len(row['term1'])

        term_2_start = sentence.find(row['term2'])
        term_2_end = term_2_start + len(row['term2'])
        start, end = (term_1_end, term_2_start) if term_2_start >= term_1_end else (term_2_end, term_1_start)

        return sentence[start:end]

    def determine_relation(self, row):
        if not pd.isnull(row['expert']):
            return 1 if row['expert'] == 1 else 0
        if not pd.isnull(row['crowd']):
            return 1 if row['crowd'] > 0 else 0
        return 1 if row['sentence_relation_score'] > 0.5 else 0

    def determine_strong_yes_relation(self, row):
        return (row['expert'] == 1 and row['crowd'] > 0 and row['sentence_relation_score'] > 0.8) or row[
            'crowd'] > 0.9 or row['sentence_relation_score'] > 0.9

    def determine_strong_no_relation(self, row):
        return (row['expert'] == -1 and row['crowd'] < 0 and row['sentence_relation_score'] < 0.2) or row[
            'crowd'] < -0.9 or row['sentence_relation_score'] < 0.1

    def add_columns(self, df, model_name):

        df['is_tested_relation'] = df.apply(lambda row: self.determine_relation(row), axis=1)
        df['relation_candidates'] = df.apply(lambda row: self.extract_relation(row), axis=1)

        df['strong_yes_relation'] = df.apply(lambda row: self.determine_strong_yes_relation(row), axis=1)
        df['strong_no_relation'] = df.apply(lambda row: self.determine_strong_no_relation(row), axis=1)

        strong_yes_df = df[df['strong_yes_relation'] == True]['relation_candidates']
        strong_no_df = df[df['strong_no_relation'] == True]['relation_candidates']

        return df, strong_yes_df, strong_no_df

class AugmentedCauseProcessor():
  def __init__(self):
    self.processor_dict = {'semeval': SemEvalProcessor('Cause-Effect(e1,e2)'), 'cause': CauseTreatProcessor(), 'ade': ADEProcessor('cause')}

  def determine_relation(self,row):
    return self.processor_dict[row['source']].determine_relation(row)

  def extract_relation(self,row):
    return self.processor_dict[row['source']].extract_relation(row)

  def determine_strong_yes_relation(self, row):
    return row['is_tested_relation']==1

  def determine_strong_no_relation(self, row):
    return  row['is_tested_relation']==0

  def add_columns(self, df, model_name):
    df['is_tested_relation']=df.apply(lambda row: self.determine_relation(row), axis=1)
    df['relation_candidates'] = df.apply(lambda row: self.extract_relation(row), axis=1)

    df['strong_yes_relation']=df.apply(lambda row: self.determine_strong_yes_relation(row), axis=1)
    df['strong_no_relation']=df.apply(lambda row: self.determine_strong_no_relation(row), axis=1)

    strong_yes_df=df[df['strong_yes_relation']==True]['relation_candidates']
    strong_no_df=df[df['strong_no_relation']==True]['relation_candidates']

    return df, strong_yes_df, strong_no_df

class SemEvalProcessor():
    def __init__(self, class_of_interest):
        self.class_of_interest = class_of_interest

    def extract_entities(self, row):
        sentence = row['sentence']
        if row['label'] == 'Cause-Effect(e1,e2)':
            e1_start_tag, e1_end_tag, e2_start_tag, e2_end_tag = 'E1_START', 'E1_END', 'E2_START', 'E2_END'
        else:
            e2_start_tag, e2_end_tag, e1_start_tag, e1_end_tag = 'E1_START', 'E1_END', 'E2_START', 'E2_END'
        term_1_start = sentence.find(e1_start_tag)
        term_1_end = sentence.find(e1_end_tag) + len(e1_end_tag)
        term1 = sentence[term_1_start: term_1_end]

        term_2_start = sentence.find(e2_start_tag)
        term_2_end = sentence.find(e2_end_tag) + len(e2_end_tag)
        term2 = sentence[term_2_start: term_2_end]
        return term1, term2, term_1_start, term_2_start, term_1_end, term_2_end

    def extract_relation(self, row, context_window=0):
        sentence = row['sentence']
        term1, term2, term_1_start, term_2_start, term_1_end, term_2_end = self.extract_entities(row)

        if len(term1) > len(term2):
            sentence = sentence.replace(term1, 'XXX').replace(term2, 'YYY')
        else:
            sentence = sentence.replace(term2, 'YYY').replace(term1, 'XXX')
        if sentence.find('XXX') == -1 or sentence.find('YYY') == -1:
            sentence = sentence.replace('XXX', 'XXXYYY').replace('YYY', 'XXXYYY')
        return sentence

        start, end = (term_1_end, term_2_start) if term_2_start >= term_1_end else (term_2_end, term_1_start)
        # start, end = (term_1_start, term_2_end) if term_2_start >= term_1_end else (term_2_start, term_1_end)

        return sentence[start: end]

    def add_columns(self, df, model_name):
        df['is_tested_relation'] = df.apply(lambda row: 1 if row['label'] == self.class_of_interest else 0, axis=1)
        df['relation_candidates'] = df.apply(lambda row: self.extract_relation(row), axis=1)
        df['term1'] = df.apply(lambda row: self.extract_entities(row)[0], axis=1)
        df['term2'] = df.apply(lambda row: self.extract_entities(row)[1], axis=1)
        strong_yes_df = df[df['label'] == self.class_of_interest]['relation_candidates']
        strong_no_df = df[df['label'] != self.class_of_interest]['relation_candidates']

        return df, strong_yes_df, strong_no_df

class ADEProcessor():
  def __init__(self, class_of_interest):
    self.class_of_interest = class_of_interest

  def extract_relation(self, row, context_window = 0):

    sentence = row['sentence']

    term1 = row['term1']
    term2 = row['term2']

    if len(term1) > len(term2):
      sentence = sentence.replace(term1, 'XXX').replace(term2, 'YYY')
    else:
      sentence = sentence.replace(term1, 'YYY').replace(term2, 'XXX')
    if sentence.find('XXX')==-1 or sentence.find('YYY')==-1:
      sentence = sentence.replace('XXX', 'XXXYYY').replace('YYY','XXXYYY')
    return sentence


    start, end = (term_1_end, term_2_start) if term_2_start >= term_1_end else (term_2_end, term_1_start)
    #start, end = (term_1_start, term_2_end) if term_2_start >= term_1_end else (term_2_start, term_1_end)

    return sentence[start: end]


  def add_columns(self, df, model_name=None):
    df['is_tested_relation']=df.apply(lambda row: 1 if row['label']==self.class_of_interest else 0, axis=1)
    df['relation_candidates'] = df.apply(lambda row: self.extract_relation(row), axis=1)

    strong_yes_df=df[df['label']==self.class_of_interest]['relation_candidates']
    strong_no_df=df[df['label']!=self.class_of_interest]['relation_candidates']

    return df, strong_yes_df, strong_no_df
