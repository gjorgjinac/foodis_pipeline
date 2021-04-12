
import os
import pandas as pd


def read_data_for_task(model_name):
    return [read_df_from_project_dir(f'{split_name}.csv', f'splits_{model_name}_balanced_False') for split_name in
            ['train', 'val', 'test']]


def save_df_in_project_dir(df, file_name, subdirectory=None):
    #create_directory_if_not_exists(get_path_from_project_dir(subdirectory))
    df.to_csv(get_path_from_project_dir(file_name, subdirectory))


def read_df_from_project_dir(file_name, subdirectory=None):
    return pd.read_csv(get_path_from_project_dir(file_name, subdirectory), index_col=[0], encoding="latin-1")


def get_path_from_project_dir(file_name, subdirectory=None):
    return os.path.join(subdirectory, file_name)


def save_report_to_file(report, vectorizer_model_name, file_name, nn_model_name='bert'):
    ret = pd.DataFrame.from_dict(report)
    ltx = ret.to_latex(label="tab:results", caption="Results")
    print(ret)
    with open(file_name, "w") as f_out:
        f_out.write(f"{ltx}\n")


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


def augment_data(og_data, data, split_name):
    data_from_other_sources = og_data[og_data['source'] != source]
    positive_samples_count = data[data['is_tested_relation'] == 1].shape[0]
    negative_samples_count = data[data['is_tested_relation'] == 0].shape[0]
    label_with_less_samples = 0 if negative_samples_count < positive_samples_count else 1
    number_of_samples_to_add = positive_samples_count - negative_samples_count if label_with_less_samples == 0 else negative_samples_count - positive_samples_count
    print(f'{split_name} split: Adding {number_of_samples_to_add} of class {label_with_less_samples}')

    data_to_append = data_from_other_sources[
        data_from_other_sources['is_tested_relation'] == label_with_less_samples].sample(number_of_samples_to_add)
    data = data.append(data_to_append).sample(frac=1)
    return data


def extract_data_from_source(primary_source, og_train_data, og_val_data, og_test_data, balance):
    train_data, val_data, test_data = [data[data['source'] == primary_source] for data in
                                       [og_train_data, og_val_data, og_test_data]]
    if balance == True:
        train_data = augment_data(og_train_data, train_data, 'train')
    return [train_data, val_data, test_data]


def print_class_distribution(data, split_name, class_name='is_tested_relation'):
    print(
        f'Class distribution in {split_name}: positives={data[data[class_name] == 1].shape[0]}  negatives={data[data[class_name] == 0].shape[0]}')


from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
from sklearn.metrics import classification_report
import numpy as np

model_name = 'cause_augmented'
source = 'ade'
print(f'source: {source}')
balance_with_sampling = True
processor = CauseTreatProcessor()
og_train_data, og_val_data, og_test_data = read_data_for_task(model_name)

if np.any([column not in og_train_data.columns for column in ['relation_candidates', 'is_tested_relation']]):
    og_train_data, og_val_data, og_test_data = [processor.add_columns(data, model_name)[0] for data in
                                                [og_train_data, og_val_data, og_test_data]]
[print_class_distribution(data, split) for (data, split) in
 [(og_train_data, 'train'), (og_val_data, 'val'), (og_test_data, 'test')]]
if model_name == 'cause_augmented':
    train_data, val_data, test_data = extract_data_from_source(source, og_train_data, og_val_data, og_test_data,
                                                               balance_with_sampling)
else:
    train_data, val_data, test_data = og_train_data, og_val_data, og_test_data

train_data, val_data, test_data = [data[['relation_candidates', 'is_tested_relation']] for data in
                                   [train_data, val_data, test_data]]

[print_class_distribution(data, split) for (data, split) in
 [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]]

train_data.columns = ["text", "labels"]
val_data.columns = ["text", "labels"]
test_data.columns = ["text", "labels"]

model_args = ClassificationArgs(num_train_epochs=10, do_lower_case=True,
                                overwrite_output_dir=True,
                                output_dir=get_path_from_project_dir(f'{model_name}_{source}'),
                                best_model_dir=get_path_from_project_dir(f'{model_name}_{source}/best'),
                                save_model_every_epoch=False, save_eval_checkpoints=False, save_steps=-1,
                                evaluate_during_training_verbose=True, evaluate_during_training=True,
                                early_stopping_consider_epochs=True, use_early_stopping=True, early_stopping_patience=5,
                                early_stopping_delta=5e-3)

model = ClassificationModel(
    "bert",
    "bert-base-uncased",
    num_labels=2
    , args=model_args)

model.train_model(train_data, eval_df=val_data)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(val_data)

# Make predictions with the model
predictions, raw_outputs = model.predict(list(test_data['text'].values))
report = classification_report(test_data['labels'], predictions, output_dict=True)
report_file_name = f"{model_name}_res.txt"
print(report)
save_report_to_file(report, vectorizer_model_name='bert', file_name=report_file_name, nn_model_name='simple_bert')

for test_source in set(og_test_data['source']):
    test_data = og_test_data.copy()
    test_data = test_data[test_data['source'] == test_source]
    test_data = test_data[['relation_candidates', 'is_tested_relation']]

    test_data.columns = ["text", "labels"]
    predictions, raw_outputs = model.predict(list(test_data['text'].values))
    report = classification_report(test_data['labels'], predictions, output_dict=True)
    print(test_source)
    report_file_name = f'reports/trained_on_{source}_cause_balanced_augmented_tested_on_{test_source}'
    save_report_to_file(report, vectorizer_model_name='bert', file_name=report_file_name, nn_model_name='simple_bert')