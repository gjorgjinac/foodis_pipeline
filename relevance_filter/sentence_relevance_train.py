from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

path_to_drive_project_folder = ''


def save_df_in_project_dir(df, file_name, subdirectory=None):
    df.to_csv(get_path_from_project_dir(file_name, subdirectory))


def read_df_from_project_dir(file_name, subdirectory=None):
    return pd.read_csv(get_path_from_project_dir(file_name, subdirectory), index_col=[0])


def get_path_from_project_dir(file_name, subdirectory=None):
    return os.path.join(subdirectory, file_name)


class_name = 'knowledge_type'
positive_values = ['Observation', 'Analysis', 'Fact']
negative_values = ['Investigation', 'Method', 'Other']

df = read_df_from_project_dir('events_KT.csv', 'data').sample(frac=1)
df = df[df['knowledge_type'] != 'Other']
print(set(df['knowledge_type'].values))
df['is_relevant'] = [1 if value in positive_values else 0 for value in df[class_name].values]
print(df[['is_relevant', class_name]].head())
class_name = 'is_relevant'
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
label_encoder = LabelEncoder()

# df = read_df_from_project_dir('balanced_events_KT.csv').sample(frac=1)

df = df[['sentence', 'is_relevant']]
df.columns = ["text", "labels"]
df['labels'] = label_encoder.fit_transform(df['labels'])
print(df.groupby('labels').count())
min_count_of_class = np.min(df.groupby('labels').count().values)
min_count_of_class -= int(0.2 * min_count_of_class)

balanced_train_df = pd.DataFrame()
for class_value in set(df['labels'].values):
    balanced_train_df = balanced_train_df.append(df[df['labels'] == class_value].sample(min_count_of_class))

print(balanced_train_df.shape)
print(df.shape)
eval_df = df.drop(balanced_train_df.index, axis=0)
eval_df, test_df = train_test_split(eval_df, test_size=0.5)

train_df = balanced_train_df

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=10, do_lower_case=True,
                                overwrite_output_dir=True,
                                output_dir=get_path_from_project_dir('sentence_relevance', 'trained_models'),
                                best_model_dir=get_path_from_project_dir('sentence_relevance/best', 'trained_models'),
                                save_model_every_epoch=False, save_eval_checkpoints=False, save_steps=-1,
                                evaluate_during_training_verbose=True, evaluate_during_training=True,
                                early_stopping_consider_epochs=True, use_early_stopping=True, early_stopping_patience=5,
                                early_stopping_delta=5e-3)
# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-uncased", args=model_args, num_labels=len(set(df['labels']))
)

# Train the model
model.train_model(train_df, eval_df=eval_df)

trained_model = ClassificationModel(
    "bert", get_path_from_project_dir('sentence_relevance')
)

result, model_outputs, wrong_predictions = trained_model.eval_model(eval_df)

# Make predictions with the model

predictions, raw_outputs = trained_model.predict(eval_df['text'].values)

print(set(predictions))
y_test = eval_df['labels']
from sklearn.metrics import accuracy_score, classification_report

print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
