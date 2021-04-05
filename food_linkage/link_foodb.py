import pandas as pd
all_candidates = pd.read_csv('../results/bert_all_candidates.csv', index_col=[0])#[['term1','entity_id_y']]

foodb = pd.read_csv('../foodb/Food.csv', index_col=[0])
foodb=foodb.rename({'public_id':'foodb_public_id'}, axis=1)

foodb_scientific = foodb.copy()
foodb_scientific = foodb_scientific[~foodb_scientific['name_scientific'].isna()]
foodb_scientific['name_scientific']=foodb_scientific['name_scientific'].apply(lambda x: x.lower())
foodb_scientific = foodb_scientific.set_index('name_scientific')[['foodb_public_id','itis_id','wikipedia_id','ncbi_taxonomy_id']]

foodb_non_scientific = foodb.copy()
foodb_non_scientific = foodb_non_scientific[~foodb_non_scientific['name'].isna()]
foodb_non_scientific['name']=foodb_non_scientific['name'].apply(lambda x: x.lower())
foodb_non_scientific = foodb_non_scientific.set_index('name')[['foodb_public_id','itis_id','wikipedia_id','ncbi_taxonomy_id']]

foodb = foodb_non_scientific.append(foodb_scientific)
all_candidates=all_candidates.merge(foodb, left_on='term1', right_on=foodb.index, how='left')

all_candidates.to_csv('results/extractors_applied.csv')
'''all_candidates['foodb']=all_candidates['term1'].apply(lambda x: foodb_scientific[x] if x in foodb_scientific.index else foodb_non_scientific[x] if x in foodb_non_scientific.index else None)
'''
exit()



foodb['name'] = foodb['name'].apply(lambda x: x.lower())
foodb['name_scientific'] = foodb['name_scientific'].fillna('').apply(lambda x: x.lower())
foodb_matches_non_scientific = all_candidates.merge(foodb, left_on='term1', right_on='name')
foodb_matches_scientific = all_candidates.merge(foodb, left_on='term1', right_on='name_scientific')
foodb_matches = foodb_matches_scientific.append(foodb_matches_non_scientific)
print(foodb_matches)
print()
print(f'unique mentions of foods linked to foodb: {foodb_matches["public_id"].dropna().shape[0]}')
print(f'unique foods linked to foodb: {foodb_matches["public_id"].dropna().drop_duplicates().shape[0]}')
print(f'unique DO-foodb pairs: {foodb_matches[["public_id", "entity_id_y"]].dropna().drop_duplicates().shape[0]}')
print('-------------------')
exit()