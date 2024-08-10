import pickle
import os
import re
import spacy
import nltk
import sys
import numpy as np
DIRNAME = os.path.dirname(__file__)

# nltk.download('wordnet')    # download the wordnet corpus for lemmatization (only once)
# nltk.download('stopwords')  # download the stopwords corpus for removing stopwords (only once)


def extract_id_and_text(query_file):
    """
    Extracts the id and text from the given file
    :param file_name: the file name to be read
    :return: string of all id and text separated b  y .I and .W
    """
    documents_dict = {}
    # regex for extracting id and text blocks
    regex = re.compile(r'(\.I|\.W)((?:(?!\.(?:I|W|B|T|A)).)+)', re.DOTALL)
    with open(query_file, 'r') as f:
        text = f.read()
        text = regex.findall(text)
        i = 0
        while i < len(text)-1:
            if text[i][0] == '.I':
                j = i + 2
                if text[i+1][1].strip() == '':
                    i += 1
                    continue
                documents_dict[int(text[i][1].strip())] = text[i+1][1].strip()
                while j < len(text) and text[j][0] != '.I':
                    documents_dict[int(text[i][1].strip())] += text[j-1][1].strip()
                    j += 1
                i = j - 1
            i += 1
    return documents_dict
            
def preprocess_text(ddict):
    """
    - Removes all the punctuations from the text
    - Removes all the stop words from the text
    - Converts all the words to lower case
    - Lemmatizes all the words
    :param document_dict: the dictionary of id and text
    :return: the dictionary of id and preprocessed text
    """
    nlp = spacy.load("en_core_web_trf")
    stemmer = nltk.stem.PorterStemmer()
    for key in ddict:
        print(key)
        ddict[key] = re.sub(r'[^\w\s]|\\n', ' ', ddict[key])
        ddict[key] = nlp(ddict[key].lower())
        ddict[key] = [stemmer.stem(word.lemma_) for word in ddict[key] if not word.is_stop]
        ddict[key] = [word for word in ddict[key] if word.strip() not in [' ', '', '\n', '\t']]
    return ddict

def save_txt_file(query_dict):
    """
    Saves the preprocessed text in a txt file
    :param query_dict: the dictionary of id and preprocessed text
    """
    with open(f"{DIRNAME}/queries_21CS10083.txt", "w") as f:
        for key in query_dict:
            f.write(f"{key}\t{' '.join(query_dict[key])}\n")
    print("Queries saved in queries.txt")
    

# Functions for each scheme element

def load_inverted_index(filename):
    with open(filename, 'rb') as f:
        inverted_index = pickle.load(f)
    return inverted_index

def logarithmic_tf(tf):
    # set non zero values to 1 + log10(tf) else 0
    zero_indices = np.argwhere(tf == 0)
    out = np.log10(tf, where=tf>0) + 1
    out[zero_indices[:, 0], zero_indices[:, 1]] = 0
    return out

def augmented_tf(tf, max_tf):
    max_tf = np.array(max_tf).reshape(-1, 1)
    return 0.5 + (0.5 * tf)/max_tf

def boolean_tf(tf):
    return np.where(tf > 0, 1, 0)

def Log_average_tf(tf, avg_tf):
    zero_indices = np.argwhere(tf == 0)
    out = np.log10(tf, where=tf>0) + 1
    out[zero_indices[:, 0], zero_indices[:, 1]] = 0
    return out/(1 + np.log10(avg_tf).reshape(-1, 1))

def idf(df, N):
    return np.log10(N/np.array(df).reshape(1, -1))

def p_idf(df, N):
    df = np.array(df).reshape(1, -1)
    out = np.log((N - df)/df)
    out[out < 0] = 0
    return out

def cosine_normalization(weights):
    sq_w = np.square(weights)
    sum_w = np.sum(sq_w, axis = 1).reshape(-1, 1)
    sqrt_w = np.sqrt(sum_w)
    return weights/sqrt_w

def apply_scheme(tf_idf, df, scheme, N):
    tf_scheme, df_scheme, normalization = list(scheme)
    if tf_scheme == 'l':    # tf = 1 + log10(tf) if tf > 0 else 0
        tf_idf[:, 1:] = logarithmic_tf(tf_idf[:,1:])
    elif tf_scheme == 'a':  # tf = 0.5 + (0.5 * tf)/max_tf
        tf_idf[:, 1:] = augmented_tf(tf_idf[:,1:], np.max(tf_idf[:,1:], axis=1))
    elif tf_scheme == 'L':  # log average tf
        non_zero_count = np.sum(tf_idf[:, 1:] > 0, axis = 1)
        avg_tf = np.sum(tf_idf[:, 1:], axis=1)/non_zero_count
        tf_idf[:, 1:] = Log_average_tf(tf_idf[:,1:], avg_tf)

    if df_scheme == 'n': # no idf weight
        tf_idf[:, 1:] *= 1
    elif df_scheme == 't':  # log(N/dft) idf weight
        tf_idf[:, 1:] *= idf(df, N)
    elif df_scheme == 'p':  # log(N-dft/dft) idf weight
        tf_idf[:, 1:] *= p_idf(df, N)
    
    # Cosine normalization
    if normalization == 'c':
        tf_idf[:, 1:] = cosine_normalization(tf_idf[:, 1:])
    return tf_idf

def ranking(doc_tf_idf, query_tf_idf, filename='A'):
    """
    Ranks the documents based on the tf-idf score
    :param doc_tf_idf: the dictionary of id and tf-idf score for documents
    :param query_tf_idf: the dictionary of id and tf-idf score for queries
    :return: the dictionary of id and tf-idf score
    """
    docs = doc_tf_idf[:, 1:]
    queries = query_tf_idf[:, 1:]
    scores = queries @ docs.T # matrix multiplication of queries and documents, gives cosine similarity as the score
    ranked_docs = np.argsort(scores, axis=1)[:, ::-1]   # sort the documents in descending order of scores
    # Write the ranked list to a file
    with open(f"Assignment2_21CS10083_ranked_list_{filename}.txt", "w") as f:
        for i in range(ranked_docs.shape[0]):
            # print(f"Query {int(query_tf_idf[i][0])} :")
            f.write(f"{int(query_tf_idf[i][0])} : ")
            for j in range(50):
                # print(f"Document {doc_tf_idf[ranked_docs[i][j]][0]}: {scores[i][ranked_docs[i][j]]}")
                f.write(f"{int(doc_tf_idf[ranked_docs[i][j]][0])} ")
            f.write('\n')

def compute_tf_idf(query_dict, document_dict, inverted_index, scheme="lnc.ltc", filename='A'):
    """
    Computes the tf-idf score for each query and document pair
    :param query_dict: the dictionary of id and preprocessed text
    :param document_dict: the dictionary of id and preprocessed text
    :param inverted_index: the inverted index dictionary
    :param scheme: the scheme for calculating tf-idf score
    :return: the dictionary of id and tf-idf score
    """
    
    vocab = inverted_index.keys()
    df = [len(inverted_index[word]) for word in vocab]
    query_tf_idf = np.zeros((len(query_dict), len(vocab)+1)) # matrix of size (no. of queries, vocab size + 1)
    document_tf_idf = np.zeros((len(document_dict), len(vocab) + 1)) # matrix of size (no. of documents, vocab size + 1)
    N = len(document_dict)  # total number of documents

    # calculating tf score for each document and query
    for i, doc_id in enumerate(document_dict.keys()):
        document_tf_idf[i][0] = doc_id
        for word in document_dict[doc_id]:
            if word in vocab:
                document_tf_idf[i][list(vocab).index(word) + 1] += 1
    
    for i, query_id in enumerate(query_dict.keys()):
        query_tf_idf[i][0] = query_id
        for word in query_dict[query_id]:
            if word in vocab:
                query_tf_idf[i][list(vocab).index(word) + 1] += 1

    doc_scheme, q_scheme = scheme.split('.')
    # Apply tf-idf scheme
    document_tf_idf = apply_scheme(document_tf_idf, df, doc_scheme, N)
    query_tf_idf = apply_scheme(query_tf_idf, df, q_scheme, N)


    ranking(document_tf_idf, query_tf_idf, filename=filename)

def main():
    DATA_FOLDER = sys.argv[1]   # CRAN Dataset folder
    INVERTED_INDEX_FILENAME = sys.argv[2]   # bin file
    DOCUMENT_DICT_DIR = os.path.join(DATA_FOLDER, "cran.all.1400")
    QUERY_DICT_DIR = os.path.join(DATA_FOLDER, "cran.qry")
    # Loading the inverted index
    inverted_index = load_inverted_index(INVERTED_INDEX_FILENAME)
    # Extract text from the files
    document_dict = extract_id_and_text(DOCUMENT_DICT_DIR)
    query_dict = extract_id_and_text(QUERY_DICT_DIR)
    # Preprocess the text with spacy
    document_dict = preprocess_text(document_dict)
    query_dict = preprocess_text(query_dict)
    # Run the ranker on 3 schemes
    compute_tf_idf(query_dict, document_dict, inverted_index, scheme="lnc.ltc", filename='A')
    compute_tf_idf(query_dict, document_dict, inverted_index, scheme="lnc.Ltc", filename='B')
    compute_tf_idf(query_dict, document_dict, inverted_index, scheme="anc.apc", filename='C')

if __name__ == "__main__":
    main()