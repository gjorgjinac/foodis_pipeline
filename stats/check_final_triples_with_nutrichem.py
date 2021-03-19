import time
import traceback

import requests
import re
import os
import subprocess
import pandas as pd
def query_nutrichem_by_disease(disease):
    cmd='''
    Invoke-WebRequest -Uri "http://147.8.185.62/services/NutriChem-2.0/cgi-bin/webface2.php" `
-Method "POST" `
-Headers @{
"Cache-Control"="max-age=0"
  "Upgrade-Insecure-Requests"="1"
  "Origin"="http://147.8.185.62"
  "User-Agent"="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.146 Safari/537.36"
  "Accept"="text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
  "Referer"="http://147.8.185.62/services/NutriChem-2.0/"
  "Accept-Encoding"="gzip, deflate"
  "Accept-Language"="en-US,en;q=0.9"
} `
-ContentType "multipart/form-data; boundary=----WebKitFormBoundaryR07awhPG4h2RJrkv" `
-Body ([System.Text.Encoding]::UTF8.GetBytes("------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"configfile`"$([char]13)$([char]10)$([char]13)$([char]10)/opt/lampp/htdocs/services/NutriChem-2.0/NutriChem.cf$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"food_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"food_search_section`"$([char]13)$([char]10)$([char]13)$([char]10)None$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"disease_name`"$([char]13)$([char]10)$([char]13)$([char]10)'''+ disease + '''$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"diseaseclass_name`"$([char]13)$([char]10)$([char]13)$([char]10)None$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"disease_search_section`"$([char]13)$([char]10)$([char]13)$([char]10)food_disease$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"compound_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"smiles_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"inchi_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"drug_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"limit`"$([char]13)$([char]10)$([char]13)$([char]10)1$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"maxEdge`"$([char]13)$([char]10)$([char]13)$([char]10)15$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv$([char]13)$([char]10)Content-Disposition: form-data; name=`"password`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryR07awhPG4h2RJrkv--$([char]13)$([char]10)")) \
| Select-Object -Expand Content
'''
    completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    response = str(completed.stdout)
    file_name = re.findall("nutrichem[0-9]+_disease.tsv",response)[0]

    headers = {
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.146 Safari/537.36',
        'Referer': f'http://147.8.185.62/services/NutriChem-1.0/cgi-bin/ListWrapper.php?file={file_name}&mode=disease-plant&limit=1',
    }

    response = requests.get(f'http://147.8.185.62/services/NutriChem-1.0/tmp/{file_name}', headers=headers, verify=False)
    return parse_nutrichem_response_by_disease(response)

def query_nutrichem_by_food(food):
    cmd='''
Invoke-WebRequest -Uri "http://147.8.185.62/services/NutriChem-2.0/cgi-bin/webface2.php" `
-Method "POST" `
-Headers @{
"Cache-Control"="max-age=0"
  "Upgrade-Insecure-Requests"="1"
  "Origin"="http://147.8.185.62"
  "User-Agent"="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.146 Safari/537.36"
  "Accept"="text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
  "Referer"="http://147.8.185.62/services/NutriChem-2.0/"
  "Accept-Encoding"="gzip, deflate"
  "Accept-Language"="en-US,en;q=0.9"
} `
-ContentType "multipart/form-data; boundary=----WebKitFormBoundaryzRnRLAwyVugiiy1z" `
-Body ([System.Text.Encoding]::UTF8.GetBytes("------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"configfile`"$([char]13)$([char]10)$([char]13)$([char]10)/opt/lampp/htdocs/services/NutriChem-2.0/NutriChem.cf$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"food_name`"$([char]13)$([char]10)$([char]13)$([char]10)Garlic$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"food_search_section`"$([char]13)$([char]10)$([char]13)$([char]10)food_disease$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"disease_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"diseaseclass_name`"$([char]13)$([char]10)$([char]13)$([char]10)None$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"disease_search_section`"$([char]13)$([char]10)$([char]13)$([char]10)None$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"compound_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"smiles_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"inchi_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"drug_name`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"limit`"$([char]13)$([char]10)$([char]13)$([char]10)1$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"maxEdge`"$([char]13)$([char]10)$([char]13)$([char]10)15$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z$([char]13)$([char]10)Content-Disposition: form-data; name=`"password`"$([char]13)$([char]10)$([char]13)$([char]10)$([char]13)$([char]10)------WebKitFormBoundaryzRnRLAwyVugiiy1z--$([char]13)$([char]10)")) \
| Select-Object -Expand Content
'''
    completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    response = str(completed.stdout)
    file_name = re.findall("nutrichem[0-9]+_compound.tsv",response)[0]

    headers = {
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.146 Safari/537.36',
        'Referer': f'http://147.8.185.62/services/NutriChem-1.0/cgi-bin/ListWrapper.php?file={file_name}&mode=disease-plant&limit=1',
    }

    response = requests.get(f'http://147.8.185.62/services/NutriChem-1.0/tmp/{file_name}', headers=headers, verify=False)
    print(response.text)
    return parse_nutrichem_response_by_food(response)


def parse_nutrichem_response_by_disease(response):
    results_string = response.text.split('\n')
    results_dict = {}
    for result_line in results_string[1:]:
        single_result = result_line.split('\t')
        if len(single_result) >= 6:
            reference = single_result[0]
            food_name = single_result[2].lower()
            relation = single_result[-1]
            if food_name not in results_dict.keys():
                results_dict[food_name] = {'cause': [], 'treat': []}
            # single result is in tsv form and contains: 'Reference','PlantID', 'PlantTag', 'DiseaseID', 'DiseaseTag'
            relation = 'cause' if relation == '2' else 'treat'  # if relation is cause
            results_dict[food_name][relation].append({'found_in': reference, 'tax_id': single_result[1]})
    return results_dict

def parse_nutrichem_response_by_food(response):
    results_string = response.text.split('\n')
    results_dict = {}
    for result_line in results_string[1:]:
        single_result = result_line.split('\t')
        if len(single_result) >= 6:
            reference = single_result[0]
            disease_name = single_result[3].lower()
            relation = single_result[-1]
            if disease_name not in results_dict.keys():
                results_dict[disease_name] = {'cause': [], 'treat': []}
            # single result is in tsv form and contains: 'Reference','PlantID', 'PlantTag', 'DiseaseID', 'DiseaseTag'
            relation = 'cause' if relation == '2' else 'treat'  # if relation is cause
            results_dict[disease_name][relation].append({'found_in': reference, 'tax_id': single_result[1]})
    return results_dict

if __name__ == '__main__':
    file_name = 'results/final_relations_linked'
    existing_df = pd.read_csv(f'{file_name}.csv').drop_duplicates()
    print(existing_df.shape)
    print(existing_df[existing_df['relation'] == 'cause'].shape)
    print(existing_df[existing_df['relation']=='treat'].shape)

    existing_df = existing_df.reset_index(drop=True)
    diseases = existing_df['term2'].drop_duplicates()
    foods = existing_df['term1'].drop_duplicates()

    number_of_unique_pairs = existing_df[['term1','term2']].drop_duplicates().shape[0]
    print(foods.shape[0])
    print(diseases.shape[0])

    total_found=0
    total_correct=0
    total_incorrect = 0
    for disease in diseases.values:
        time.sleep(2)
        relevant = existing_df[existing_df['term2']==disease]
        try:
            nutrichem_findings = query_nutrichem_by_disease(disease)
            for index, row in relevant.iterrows():
                food = row['term1']
                if food in nutrichem_findings.keys():
                    found_cause_evidence = len(nutrichem_findings[food]['cause'])
                    found_treat_evidence = len(nutrichem_findings[food]['treat'])
                    existing_df.loc[index, 'nutrichem_cause_found_in'] = '_'.join([f['found_in'] for f in nutrichem_findings[food]['cause']])
                    existing_df.loc[index, 'nutrichem_treat_found_in'] = '_'.join([f['found_in'] for f in nutrichem_findings[food]['treat']])
                    existing_df.loc[index, 'nutrichem_cause'] = found_cause_evidence/(found_cause_evidence+found_treat_evidence)
                    existing_df.loc[index, 'nutrichem_treat'] = found_treat_evidence/(found_cause_evidence+found_treat_evidence)
                    print(f'found: {food} {disease}')
                    rel_nutrichem = 'cause' if found_cause_evidence > found_treat_evidence else 'treat'
                    rel_foodis = row['relation']
                    total_found += 1
                    if found_cause_evidence != found_treat_evidence:
                        if rel_foodis==rel_nutrichem:
                            total_correct+=1
                            print('correct')
                            existing_df.loc[index, 'nutrichem_confirm'] = 1
                        else:
                            total_incorrect += 1
                            print('incorrect')
                            existing_df.loc[index, 'nutrichem_confirm'] = -1
                        print('================================')
            existing_df.to_csv(f'{file_name}_checked.csv')
        except Exception:
            traceback.print_exc()
            continue
    print(total_found)
    print(total_correct)
    print(total_incorrect)



