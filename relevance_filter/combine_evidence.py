from utils import read_from_dataset_dir, write_to_dataset_dir
import pandas as pd

def append_evidence_for_relation(relation_df, evidence_df, relation):
    for (term1, term2) in relation_df.index:
        evidence_for_relation = evidence_df[(evidence_df['term1']==term1) & (evidence_df['term2']==term2) & (evidence_df[f'is_{relation}']==1)]
        relation_df.loc[(term1, term2), 'evidence'] = '___'.join(evidence_for_relation['sentence'].values)
    return relation_df


def combine_evidence(dataset,
                     min_positive_classifier_support,
                     max_negative_classifier_support,
                     min_positive_evidence,
                     max_negative_evidence):

    all_candidates = read_from_dataset_dir('extractors_applied.csv', dataset)
    all_candidates=all_candidates.drop_duplicates()
    cause_columns = list(filter(lambda x: x.find('bert') > -1 and x.find('cause') > -1, all_candidates.columns))
    treat_columns = list(filter(lambda x: x.find('bert') > -1 and x.find('treat') > -1, all_candidates.columns))
    all_candidates['cause_sum'] = all_candidates[cause_columns].sum(axis=1)
    all_candidates['treat_sum'] = all_candidates[treat_columns].sum(axis=1)

    all_candidates['is_cause'] = all_candidates.apply(lambda row: 1 if (row['cause_sum'] >= min_positive_classifier_support and row['treat_sum'] <= max_negative_classifier_support )else 0, axis=1)
    all_candidates['is_treat'] = all_candidates.apply(
        lambda row: 1 if row['treat_sum'] >= min_positive_classifier_support and row['cause_sum'] <= max_negative_classifier_support else 0, axis=1)

    evidence_for_cause_count = all_candidates[all_candidates["is_cause"] == 1].shape[0]
    evidence_for_treat_count = all_candidates[all_candidates["is_treat"] == 1].shape[0]

    print(f'evidence sentences for cause: {evidence_for_cause_count}')
    print(f'evidence sentences for treat: {evidence_for_treat_count}')

    relations = all_candidates.groupby(['term1', 'term2']).sum()
    cause_relations = relations[(relations['is_treat'] <= max_negative_evidence) & (relations['is_cause'] > min_positive_evidence)]
    treat_relations = relations[(relations['is_cause'] <= max_negative_evidence) & (relations['is_treat'] > min_positive_evidence)]

    cause_relations = append_evidence_for_relation(cause_relations, all_candidates, 'cause')
    treat_relations = append_evidence_for_relation(treat_relations, all_candidates, 'treat')
    write_to_dataset_dir(cause_relations, 'cause_relations.csv',dataset)
    write_to_dataset_dir(treat_relations, 'treat_relations.csv', dataset)



