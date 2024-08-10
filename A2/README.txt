21CS10083
YASH SIRVI
INFORMATION RETREIVAL CS60092 (2023)
ASSIGNMENT 1

# Environment Information

- python version: 3.11.4
- requirements:
    - nltk==3.8.1
    - spacy==3.6.1
    - regex
    - pickle
    - python -m spacy download en_core_web_trf

# Usage

- Assignment2_21CS10083_ranker.py
    python Assignment2_21CS10083_ranker.py <path_to_data_folder> <path_to_bin_file>
    - outputs:
        - Assignment1_21CS10083_ranked_list_A.txt
        - Assignment1_21CS10083_ranked_list_B.txt
        - Assignment1_21CS10083_ranked_list_C.txt

- Assignment2_21CS10083_evaluator.py
    python Assignment1_21CS10083_search.py <path to cran.qry>
    - outputs:
        - Assignment2_21CS10083_metrics_A.txt
        - Assignment2_21CS10083_metrics_B.txt
        - Assignment2_21CS10083_metrics_C.txt

# Implementation Details 

## Assignment2_21CS10083_ranker.py

- Preprocessing is the same as Assignment 1
- There are different functions for each scheme
- I first read the processed bin file and the corresponding cran files
- I then create a matrix of tf-idf scores for each document and query separately
- I then do matrix multiplication to get the scores for each document for each query
- I save the top 50 documents for each query in a file
- I repeat this for all 3 schemes 
    - A: lnc.ltc 
    - B: lnc.Ltc
    - C: anc.apc

## Assignment2_21CS10083_evaluator.py

- Iterate over all queries
- Get the relevant documents for each query
- Calculate all metrics, append them to lists
- Write metrics of each file along with mean of all metrics to a file