
import argparse

from full_pipeline import run_full_foodis_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("-st", "--search_terms", help="The search terms used to search for PubMed articles")
parser.add_argument("-mnpcs", "--min_positive_classifier_support", default= 3, type=int, help="The minimum number of Relation Extraction classifiers that need to agree that a piece of evidence expresses a relation")
parser.add_argument("-mxncs", "--max_negative_classifier_support", default=1, type=int,help="The maximum number of Relation Extraction classifiers that can agree that a piece of evidence expresses an opposite relation")
parser.add_argument("-mnpe", "--min_positive_evidence", default=1, type=int,help="The minimum number of pieces of evidence that needs to support the existence of a relation")
parser.add_argument("-mxne", "--max_negative_evidence", default=0,type=int, help="The maximum number of pieces of evidence that can support the existence of an opposite relation")
parser.add_argument("-na", "--number_of_abstracts", default=100, type=int,help="The maximum number of abstracts that are going to be retrieved from PubMed and processed by the pipeline")

args = parser.parse_args()

if args.search_terms:
    print(f"Running FooDis pipeline for search terms '{args.search_terms}'")
    args.search_terms = args.search_terms.split(';')
    print(vars(args))
    run_full_foodis_pipeline(**vars(args))
else:
    print(f"Please specify search terms using the argument --search terms 'search terms value' or -st 'search terms value'")

