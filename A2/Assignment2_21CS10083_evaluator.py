import numpy as np
import os

# Calculate the average precision at k
def avg_precision_at(ranking, rel_docs, k):
    rel_docs = set(rel_docs)
    precision_at_k = [] # list of precision at each relevant doc position
    count = 0
    for i in range(1, k+1):
        if ranking[i-1] in rel_docs:
            count += 1
            precision_at_k.append(count/i)  # appened only if relevant doc is found
    # print(ranking)
    # print(sorted(list(rel_docs)))
    if len(precision_at_k) == 0:   # if no relevant docs found, return 0
        return 0
    return np.mean(precision_at_k)  # else return the mean of precision at each relevant doc position -> AP@k

# Calculate the normalized discounted cumulative gain at k
def ndcg_at(ranking, rel_docs_mat, k):
    rel_docs = rel_docs_mat[:,0]
    rel_scores = rel_docs_mat[:,1]
    rel_docs_set = set(rel_docs)
    # Get scores for the retrieved documents
    ranked_scores = [rel_scores[rel_docs == doc][0] if doc in rel_docs_set else 0 for doc in ranking[:20]]
    # start calculating dcg and idcg
    dcg_at_k = [ ranked_scores[0] ]
    for i in range(1, 20):
        dcg_at_k.append(dcg_at_k[-1] + ranked_scores[i]/np.log2(i+1))
    # for ideal dcg, sort the scores in descending order -> best possible order
    ranked_scores = sorted(ranked_scores, reverse=True)
    idcg_at_k = [ ranked_scores[0] ]
    if idcg_at_k[0] == 0:
        return 0
    for i in range(1, 20):
        idcg_at_k.append(idcg_at_k[-1] + ranked_scores[i]/np.log2(i+1))
    # print(dcg_at_k[-1],"\n", idcg_at_k[-1])
    ndcg_at_k = [ dcg_at_k[i]/idcg_at_k[i] for i in range(k) ] # calculate ndcg at each position
    return ndcg_at_k[k-1]   # return the ndcg at k

# Function to read the cranqrel file and return a dictionary of relevant documents for each query
def get_relevant_docs(filename):
    rel_docs = {}
    with open(filename, 'r') as f:
        rel_docs_file = f.readlines()
    for line in range(len(rel_docs_file)):
        query, doc, rel = rel_docs_file[line].split()
        if query not in rel_docs:
            rel_docs[query] = []
        rel_docs[query] += [[int(doc), int(rel)]] if int(rel) > 0 else []
    
    for key in rel_docs:
        rel_docs[key] = np.array(sorted(rel_docs[key], key=lambda x: x[1], reverse=True)) # sort the docs
    return rel_docs 

# Function to read the ranked list file and return a dictionary of ranked documents for each query
def read_ranking(filename):
    ranking_dict = {}
    query_idx = {}
    with open(filename, 'r') as f:
        ranking_file = f.readlines()
    for line in range(len(ranking_file)):
        split_line = ranking_file[line].split(':')
        query = int(split_line[0])
        ranking = np.array(split_line[1].split()[:20], dtype=int)
        ranking_dict[query] = ranking
        query_idx[query] = line+1
    return ranking_dict, query_idx

def main():
    STANDARD_LIST_DIR = os.sys.argv[1]
    RANKED_LIST_PATH = os.sys.argv[2]
    ranking_dict, query_idx = read_ranking(RANKED_LIST_PATH)
    avg_prec_at_10 = []
    avg_prec_at_20 = []
    ndcg_at_10 = []
    ndcg_at_20 = []

    standard = get_relevant_docs(STANDARD_LIST_DIR)
    for query in ranking_dict:
        avg_prec_at_10.append(avg_precision_at(ranking_dict[query], standard[str(query_idx[query])][:,0], 10))
        avg_prec_at_20.append(avg_precision_at(ranking_dict[query], standard[str(query_idx[query])][:,0], 20))
        ndcg_at_10.append(ndcg_at(ranking_dict[query], standard[str(query_idx[query])], 10))
        ndcg_at_20.append(ndcg_at(ranking_dict[query], standard[str(query_idx[query])], 20))
        # break
    AMP_10 = np.mean(avg_prec_at_10)
    AMP_20 = np.mean(avg_prec_at_20)
    NDCG_10 = np.mean(ndcg_at_10)
    NDCG_20 = np.mean(ndcg_at_20)

    print("Mean Average Precision at 10: ", AMP_10)
    print("Mean Average Precision at 20: ", AMP_20)
    print("Mean NDCG at 10: ", NDCG_10)
    print("Mean NDCG at 20: ", NDCG_20)

    k = RANKED_LIST_PATH.split('_')[-1].split('.')[0]
    with open(f"Assignment2_21CS10083_metrics_{k}.txt", 'w') as f:
        f.write("Mean Average Precision at 10: " + str(AMP_10) + "\n")
        f.write("Mean Average Precision at 20: " + str(AMP_20) + "\n")
        f.write("Mean NDCG at 10: " + str(np.mean(ndcg_at_10)) + "\n")
        f.write("Mean NDCG at 20: " + str(np.mean(ndcg_at_20)) + "\n")
        space = 10
        f.write("Query".ljust(space) + "AP@10".ljust(space) + "AP@20".ljust(space) + "NDCG@10".ljust(space) + "NDCG@20".ljust(space) + "\n")
        # round off to 4 decimal places
        for query in ranking_dict:
            f.write(str(query).ljust(space) + str(round(avg_precision_at(ranking_dict[query], standard[str(query_idx[query])][:,0], 10), 4)).ljust(space) + str(round(avg_precision_at(ranking_dict[query], standard[str(query_idx[query])][:,0], 20), 4)).ljust(space) + str(round(ndcg_at(ranking_dict[query], standard[str(query_idx[query])], 10), 4)).ljust(space) + str(round(ndcg_at(ranking_dict[query], standard[str(query_idx[query])], 20), 4)).ljust(space) + "\n")
        


if __name__ == "__main__":
    main()