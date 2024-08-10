import sys
import pickle
import pprint
pp = pprint.PrettyPrinter(width=41, compact=True)

def merge_routine(posting_lists):
    """
    Merges the given posting lists by AND operation
    :param posting_lists: the posting lists to be merged
    :return: the merged posting list
    """
    result = []
    counters = [0] * len(posting_lists)
    prev_counters = [0] * len(posting_lists)
    while any([counters[i] < len(posting_lists[i]) - 1 for i in range(len(posting_lists))]):
        max_val = max([posting_lists[i][counters[i]] for i in range(len(posting_lists))])
        
        if all([posting_lists[i][counters[i]] == max_val for i in range(len(posting_lists))]):
            result.append(max_val)
            counters = [counters[i] + 1  if counters[i] < len(posting_lists[i]) - 1 else counters[i] for i in range(len(posting_lists))]
        else:
            for i in range(len(posting_lists)):
                while counters[i] < len(posting_lists[i])-1 and posting_lists[i][counters[i]] < max_val:
                    counters[i] += 1
        if counters == prev_counters:
            break
        prev_counters = counters.copy()
    return result

    # use set intersection
    # result = posting_lists[0]
    # for i in range(len(posting_lists)):
    #     result = list(set(result).intersection(set(posting_lists[i])))
    # # print(result)
    return result

def run_query(query, inverted_index):
    """d
    Runs the given query on the inverted index
    :param query: the query to be run
    :param inverted_index: the inverted index
    :return: the list of documents that satisfy the query
    """
    query = query.split()[1:]
    posting_lists = []
    for word in query:
        if word in inverted_index:
            posting_lists.append(inverted_index[word])
        else:
            return []
    return merge_routine(posting_lists)

def main():
    MODEL_PATH = sys.argv[1]
    QUERY_FILE = sys.argv[2]
    inverted_index = pickle.load(open(MODEL_PATH, "rb"))
    query_txt = open(QUERY_FILE, "r").readlines()
    n = 0
    with open("Assignment1_21CS10083_output.txt", "w") as f:
        for query in query_txt:
            query = query.strip()
            f.write(f"{query.split()[0]}: ")
            result = run_query(query, inverted_index)
            
            if not len(result) == 0:
                n += 1
                for doc_id in result:
                    f.write(f"{doc_id} ")
            f.write("\n")

if __name__ == "__main__":
    main()