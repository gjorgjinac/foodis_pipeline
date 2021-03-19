import pandas as pd
df = pd.read_csv('../foodb/Food.csv', index_col=[0])
print(df.shape)
for column in ['ncbi_taxonomy_id','itis_id', 'wikipedia_id' ]:
    print(column)
    print(df[column].dropna().drop_duplicates().shape[0])
    print()