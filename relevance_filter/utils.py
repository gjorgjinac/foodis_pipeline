import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path_to_drive_project_folder=''
def one_hot_encode(x, string_labels=False):
  if string_labels:
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(x)
  else:
    integer_encoded = x
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  return onehot_encoded

def example_to_features(input_ids,attention_masks,label_ids):
  return {"input_ids": input_ids, "attention_mask": attention_masks}, label_ids

def get_attention_masks(padded_matrix, padding_value = 0):
  return [[float(i != 0) for i in ii] for ii in padded_matrix]


def save_df_in_project_dir(df, file_name, subdirectory=None):
  df.to_csv(get_path_from_project_dir(file_name, subdirectory))

def read_df_from_project_dir(file_name, subdirectory = None):
  return pd.read_csv(get_path_from_project_dir(file_name, subdirectory), index_col=[0])

def get_path_from_project_dir(file_name, subdirectory = None):
  whole_directory = os.path.join(path_to_drive_project_folder, subdirectory) if subdirectory is not None else path_to_drive_project_folder
  return os.path.join(whole_directory, file_name)


def generate_splits(model_name, splits_directory, models):
    df, strong_yes_df, strong_no_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for file in models[model_name]['files']:
        file_df = read_df_from_project_dir(file['file_name'])
        file_df = file['processor'].add_columns(file_df, model_name)[0]
        print(file)
        print(file_df.columns)
        df = df.append(file_df)

    save_df_in_project_dir(df, 'all.csv', subdirectory=f'splits_{model_name}')
    df_yes = df[df['is_tested_relation'] == 1]
    df_no = df[df['is_tested_relation'] == 0].sample(n=df_yes.shape[0])
    assert df_yes.shape == df_no.shape
    balanced_df = df_yes.append(df_no)

    train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['is_tested_relation'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['is_tested_relation'])
    if not os.path.isdir(os.path.join(path_to_drive_project_folder, splits_directory)):
        os.makedirs(os.path.join(path_to_drive_project_folder, splits_directory))

    save_df_in_project_dir(train_df, 'train.csv', subdirectory=splits_directory)
    save_df_in_project_dir(val_df, 'val.csv', subdirectory=splits_directory)
    save_df_in_project_dir(test_df, 'test.csv', subdirectory=splits_directory)

