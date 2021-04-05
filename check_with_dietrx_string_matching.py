import json
import time

import pandas as pd
import requests

from dietrx_foods import dietrx_foods

treat_relations_to_check_df = pd.read_csv('results/treat_relations.csv')
treat_relations_to_check_df['relation']='treat'
cause_relations_to_check_df = pd.read_csv('results/cause_relations.csv')
cause_relations_to_check_df['relation']='cause'
relations_to_check = treat_relations_to_check_df.append(cause_relations_to_check_df)
relations_to_check['term1']=relations_to_check['term1'].apply(lambda x:x.lower())
relations_to_check['term2']=relations_to_check['term2'].apply(lambda x:x.lower())


print(relations_to_check.shape)
correct = 0
incorrect = 0
matched = 0

food_names = relations_to_check['term1'].drop_duplicates().values
total_food_names = len(food_names)
for i in range(0, len(food_names)):
    print(f'{i}/{total_food_names}')
    food_name = food_names[i]
    food_in_dietrx = list(filter(lambda x: x[2].lower()==food_name or x[3].lower()==food_name, dietrx_foods))
    if len(food_in_dietrx)==0:
        print('not found in dietrx')
        continue
    food_in_dietrx = food_in_dietrx[0]

    url = f"https://cosylab.iiitd.edu.in{food_in_dietrx[4]}"
    html = requests.get(url).text
    time.sleep(0.2)
    data_start = html.find("var data = [{")
    relations_with_food = relations_to_check[relations_to_check['term1'] == food_name]
    if data_start > -1:
        data_html = html[data_start+11:]
        data_end = data_html.find("];")
        data_html = data_html[:data_end+1]
        diseases = json.loads(data_html)
        for disease in diseases:
            d = disease['(<span style="color: green">Positive</span>, <span style="color: red">Negative</span>, Chemical) Associations']
            disease_name = disease['Disease Name']['data']
            disease_id = disease['Disease ID']['data']
            foodis_relation = relations_with_food[relations_with_food['term2']== disease_name.lower()]

            if foodis_relation.shape[0] > 0:
                foodis_relation = foodis_relation['relation'].values[0]
                matched+=1
                if d['negative']!=d['positive']:
                    dietrx_relation = 'cause' if d['negative'] > d['positive'] else 'treat'
                    if foodis_relation==dietrx_relation:
                        correct+=1
                    else:
                        incorrect+=1
                    print(f'MATCHED: {matched}')
                    print(f'CORRECT: {correct}')
                    print(f'INCORRECT: {incorrect}')
                    print(f'CORRECT RATIO: {round(correct/(correct+incorrect),2)}')
                    print()

print(f'MATCHED: {matched}')
print(f'CORRECT: {correct}')
print(f'INCORRECT: {incorrect}')
