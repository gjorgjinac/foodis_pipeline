import pandas as pd

from utils import save_as_latex_table

output_dir = '../results'
mappings = {
'foodon':'FoodOn','snomedct': 'SNOMED CT','hansardClosest': 'Hansard Closest','hansardParent': 'Hansard Parent',
    'foodb_public_id':'FooDB','itis_id':'ITIS','wikipedia_id': 'Wikipedia','ncbi_taxonomy_id':'NCBI taxonomy',
'entity_id_y':'DO','snomedct_disease':'SNOMED CT','umls_disease':'UMLS','nci_disease':'NCIt','omim_disease':'OMIM',
               'efo_disease':'EFO','mesh_disease':'MESH'}
all_candidates = pd.read_csv(f'{output_dir}/bert_all_candidates.csv', index_col=[0])
all_candidates['term1']=all_candidates['term1'].apply(lambda x: x.lower())
all_candidates['term2']=all_candidates['term2'].apply(lambda x: x.lower())
cause_columns = list(filter(lambda x: x.find('bert')>-1 and x.find('cause')>-1, all_candidates.columns))
treat_columns = list(filter(lambda x: x.find('bert')>-1 and x.find('treat')>-1, all_candidates.columns))
all_candidates['cause_sum']=all_candidates[cause_columns].sum(axis=1)
all_candidates['treat_sum']=all_candidates[treat_columns].sum(axis=1)


all_candidates['is_cause']=all_candidates.apply(lambda row: 1 if row['cause_sum']>=3 and row['treat_sum'] <= 1 else 0, axis=1)
all_candidates['is_treat']=all_candidates.apply(lambda row: 1 if row['treat_sum']>=3 and row['cause_sum'] <= 1 else 0, axis=1)

evidence_for_cause_count = all_candidates[all_candidates["is_cause"]==1].shape[0]
evidence_for_treat_count = all_candidates[all_candidates["is_treat"]==1].shape[0]

print(f'evidence sentences for cause: {evidence_for_cause_count}')
print(f'evidence sentences for treat: {evidence_for_treat_count}')

relations = all_candidates.groupby(['term1','term2'] ).sum()
cause_relations = relations[(relations['is_treat']==0) & (relations['is_cause']>1)]
treat_relations = relations[(relations['is_cause']==0) & (relations['is_treat']>1)]

print(f'cause relations between unique entity mentions: {cause_relations.shape[0]}')
print(f'treat relations between unique entity mentions: {treat_relations.shape[0]}')
print(f'avg evidence for cause: {evidence_for_cause_count/cause_relations.shape[0]}')
print(f'avg evidence for treat: {evidence_for_treat_count/treat_relations.shape[0]}')
cause_relations['relation']='cause'
treat_relations['relation']='treat'
final_relations = cause_relations.append(treat_relations)
final_relations = final_relations.reset_index()

print(f'total relations between unique entity mentions: {final_relations.shape[0]}')
print(f'unique food mentions in final relations: {final_relations["term1"].drop_duplicates().shape[0]}')
print(f'unique disease mentions in final relations: {final_relations["term2"].drop_duplicates().shape[0]}')
print(f'unique food-disease pair mentions in final relations: {final_relations[["term1", "term2"]].drop_duplicates().shape[0]}')
print()


cause_df=pd.DataFrame()
treat_df=pd.DataFrame()

print(all_candidates.columns)
for food_resource in ['foodon','snomedct','hansardClosest','hansardParent','foodb_public_id','itis_id','wikipedia_id','ncbi_taxonomy_id']:
    for disease_resource in ['entity_id_y', 'snomedct_disease','umls_disease','nci_disease','omim_disease','efo_disease','mesh_disease']:
        relations = all_candidates[['term1', 'term2', food_resource, disease_resource,'is_cause','is_treat']].dropna().groupby(['term1', 'term2', food_resource, disease_resource]).sum()
        cause_relations = relations[(relations['is_treat'] == 0) & (relations['is_cause'] > 1)]
        treat_relations = relations[(relations['is_cause'] == 0) & (relations['is_treat'] > 1)]
        cause_df.loc[mappings[food_resource],mappings[disease_resource]]=int(cause_relations.shape[0])
        treat_df.loc[mappings[food_resource], mappings[disease_resource]] = int(treat_relations.shape[0])
        print(food_resource)
        print(disease_resource)
        print(cause_relations.shape)
        print(treat_relations.shape)
        print('-------------------------')
        cause_relations['relation'] = 'cause'
        treat_relations['relation'] = 'treat'

    print(cause_df)
    print(treat_df)

save_as_latex_table(cause_df, f'{output_dir}/linked_cause_food_disease_final')
save_as_latex_table(treat_df, f'{output_dir}/linked_treat_food_disease_final')

