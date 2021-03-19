import os
import pandas as pd
from config import global_output_directory_name

all_candidates = pd.DataFrame()

all_datasets = os.listdir(global_output_directory_name)
datasets_to_include = all_datasets #list(filter(lambda x: x.find('health_effects') > -1, all_datasets))
for dataset in datasets_to_include:
    file_name = os.path.join(global_output_directory_name, dataset, 'relation_candidates.csv')
    if os.path.isfile(file_name):
        dataset_candidates = pd.read_csv(file_name, index_col=[0])
        dataset_candidates = dataset_candidates[((dataset_candidates['support']>2) |
                                                (~dataset_candidates['snomedct'].isna()) |
                                                (~dataset_candidates['hansard'].isna()) |
                                                (~dataset_candidates['hansardParent'].isna()) |
                                                (~dataset_candidates['hansardClosest'].isna()) |
                                                (~dataset_candidates['foodon'].isna()) |
                                                (dataset_candidates['extractor_x']=='foodb_scientific'))
            &(~dataset_candidates['sentence_x'].isna())
        ] #[~dataset_candidates['entity_id_y'].isna()]
        all_candidates=all_candidates.append(dataset_candidates, ignore_index=True)
print(all_candidates.index.is_unique)
all_candidates=all_candidates.rename(columns={'sentence_y':'sentence'})

all_candidates=all_candidates
print(all_candidates.shape[0])
pairs_count = all_candidates.groupby(['text_x','text_y']).count()
print(pairs_count[pairs_count>1].shape[0])
all_candidates.to_csv(os.path.join(f'{global_output_directory_name}/all_relation_candidates.csv'))