# The FooDis pipeline

The FooDis pipeline enables mining abstracts of biomedical scientific papers from Pubmed, in order to identify relations between _**food**_ and _**disease**_ entities. The entities can be associated with a _**cause**_ or a _**treat**_ relation, or there can be no association between them.
## Pipeline overview

The initial step is querying PubMed and retrieving abstracts of scientific papers. This is done by searching PubMed using a set of search terms which are provided as input arguments to the pipeline.

Different Named Entity Recognition (NER) and Named Entity Linking(NEL) methods are applied to the text of the retrieved paper abstracts for the extraction of food and disease entities, which are linked to several existing resources from the biomedical domain and the domain of food and nutrition. 
In particular, the SABER tool is used to extract disease entities and link them to the Disease Ontology, which can be used to further link the entity to SNOMED CT, UMLS, NCIT, OMIM, EFO and MESH.
For the extraction of food entities, we use the BuTTER and FoodNER methods, as well as two methods based on simple string matching of concepts from FooDB. FoodNER can link the extracted food concepts to the FoodOn and SNOMED CT ontologies, and the Hansard Corpus. The two extractors which use the concepts in the FooDB can link the extracted entities to Wikipedia articles, ITIS and NCBIT. The results of these 4 extractors are combined using a custom voting scheme in order to generate the final food entities

Next, sentences which express facts or analysis of the research and contain at least one pair of food and disease entities are extracted. The sentences are annotated for the existence of a _cause_ or _treat_ relation by 4 different classifiers per relation.
The annotations of the 8 classifiers are combined using a voting strategy to determine whether a sentence expresses a _cause_ or a _treat_ relation. 
In order a relation to be accepted as positive, at least X out of the 4 classifiers that identify the relation need to generate a positive prediction, and a maximum of Y classifier that identifies the opposite relation is allowed to generate a positive prediction. Here, X and Y can be given as parameters to the pipeline, using the _min_positive_classifier_support_ and _max_negative_classifier_support_ command line arguments.

We refer to each sentence where a _cause_ or _treat_ relation is identified between a food-disease pair as a piece of evidence supporting that relation. All of the pieces of evidence found for a specific (food, relation, disease) triple are combined in order to determine whether the relation triple is valid. We consider the triple to be valid only if there is at least M pieces of evidence that support that relation and there is N pieces of evidence for the existence of the opposite relation. Here, M and N are parameters given to the pipeline. They can be set using the arguments _min_positive_evidence_ and _max_negative_evidence_, respectively.
![FooDis pipeline](images/pipeline.png "Initial foodis pipeline")

## Setup
Python 3.7 or 3.8 is recommended to be used.

All of the required libraries are listed in the _requirements.txt_ file and can be installed using the following command:

- `cat requirements.txt | xargs -n 1 pip install` (Linux)

- `FOR /F %k in (requirements.txt) DO pip install %k` (Windows)

After the requirements are installed, run the following line to download the needed spacy model:
`python -m spacy download en_core_web_sm`

Download the pretrained models from this link https://portal.ijs.si/nextcloud/s/mrRrk2sGNiwBLe2 and place them in the trained_models subdirectory

## Usage
### Running the pipeline
The pipeline can be run using the _foodis.py_ script, where the following arguments can be specified:
  - _st_ or _search_terms_ - string containing search terms separated by ";" - The search terms used to search for PubMed articles
  - _na_ or _number_of_abstracts_ - integer in the range [1, infinity) - default 100 - The maximum number of abstracts that are going to be retrieved from PubMed and processed by the pipeline
  - _mnpcs_ or _min_positive_classifier_support_ - integer in the range [1,4] - default 3 - The minimum number of Relation Extraction classifiers that need to agree that a piece of evidence expresses a relation, for a piece of evidence to be considered valid
  - _mxncs_ or _max_negative_classifier_support_ - integer in the range [0,4] - default 1 - The maximum number of Relation Extraction classifiers that can agree that a piece of evidence expresses an opposite relation, for a piece of evidence to be considered valid
  - _mnpe_ or _min_positive_evidence_ - integer in the range [1, infinity) - default 1 - The minimum number of pieces of evidence that needs to support the existence of a relation, for a relation to be considered valid
  - _mxne_ or _max_negative_evidence_ - integer in the range [0, infinity) - default 0 - The maximum number of pieces of evidence that can support the existence of an opposite relation, for a relation to be considered valid

Only the _search_terms_ argument is required.
These arguments refer to a single run of the pipeline. In the _config.py_ file, one can set the directory where the outputs files are going to be generated, and specify whether a GPU should be used.

### Generated results
The results for each search term are saved in a dedicated subdirectory of the specified output directory.
Apart from the final extracted relations, which are saved in the _cause_relations.csv_ and _treat_relations.csv_ files, the pipeline saves the results of several intermediate steps, into the following files:
- Outputs of the **Search Pubmed** component:
    - _abstracts.csv_ - The retrieved abstracts for the specified search words
- Outputs of the **Disease NER & NEL** component:    
    - _saber_diso.csv_ - The extracted disease entities from the SABER model
- Outputs of the **Food NER & NEL** component:
    - _butter.csv_, _foodner_web.csv_, _foodb_scientific_False_non_scientific_True.csv_, _foodb_scientific_True_non_scientific_False.csv_ - The food entities extracted by the BuTTER, FoodNER, FooDB Non-Scientific and FooDB Scientific extractors, respectively
    - _foods_support_x.csv_ - The combined results of the food extractors, where x is the number of extractors which need to confirm the existence of a food entity for it to be considered valid
- Outputs of the **Sentence relevance filtering** component:
    - _relation_candidates.csv_ - The extracted sentences which contain at least one food-disease pair
    - _relevant_candidates.csv_ - The subset of the sentences which contain at least one food-disease pair and express a _Fact_ or _Analysis_, and are used to extract the pieces of evidence for each relation
    - _irrelevant_candidates.csv_ - The subset of the sentences which contain at least one food-disease pair and express a _Hypothesis_, _Method_ or _Other_, and are discarded by the sentence relevance filter
    - _sentence_relevance_values.csv_ - Union of the _relevant_candidates.csv_, and _irrelevant_candidates.csv_, with an assigned binary indicator of whether each sentence was deemed to be relevant or not, by the sentence relevance filter component
- Outputs of the **Relation classification** component:
    - _extractors_applied.csv_ - The relevant sentences, with assigned predictions of each of the 8 classifiers which detect the _cause_ or _treat_ relation
- Outputs of the **Final relation determination** component:   
    - _cause_relations.csv_, _treat_relations.csv_ - The final extracted relations
