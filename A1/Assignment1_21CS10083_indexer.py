import os
import re
import pickle
import nltk
import sys
import spacy
DIRNAME = os.path.dirname(__file__)

nltk.download('wordnet')
nltk.download('stopwords')

def extract_id_and_text(cran_folder):
    """
    Extracts the id and text from the given file
    :param file_name: the file name to be read
    :return: string of all id and text separated by .I and .W
    """
    file_name = f"{cran_folder}/cran.all.1400"
    documents_dict = {}
    regex = re.compile(r'(\.I|\.W)((?:(?!\.(?:I|W|B|T|A)).)+)', re.DOTALL)
    with open(file_name, 'r') as f:
        text = f.read()
        text = regex.findall(text)
        i = 0
        while i < len(text)-1:
            if text[i][0] == '.I':
                j = i + 2
                documents_dict[int(text[i][1].strip())] = text[i+1][1].strip()
                while j < len(text) and text[j][0] != '.I':
                    documents_dict[int(text[i][1].strip())] += text[j-1][1].strip()
                    j += 1
                i = j - 1
            i += 1
    return documents_dict
            
def preprocess_text(document_dict):
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
    for key in document_dict:
        print("Processing docID: ", key)
        document_dict[key] = re.sub(r'[^\w\s]', ' ', document_dict[key])
        document_dict[key] = nlp(document_dict[key].lower())
        document_dict[key] = [stemmer.stem(word.lemma_) for word in document_dict[key] if not word.is_stop]
        # remove empty strings and new line characters
        document_dict[key] = [word for word in document_dict[key] if word != '' and word != '\n']
    return document_dict
    
def create_inverted_index(document_dict):
    """
    Creates the inverted index from the given dictionary
    :param document_dict: the dictionary of id and preprocessed text
    :return: the inverted index
    """
    inverted_index = {}
    for key in document_dict:
        for word in document_dict[key]:
            if word not in inverted_index:
                inverted_index[word] = [key]
            else:
                if key not in inverted_index[word]:
                    inverted_index[word].append(key)
        # sort the posting list in ascending order
        if key in inverted_index:
            inverted_index[key].sort()
    return inverted_index

def main():
    CRAN_FOLDER = sys.argv[1]
    document_dict = extract_id_and_text(CRAN_FOLDER)
    document_dict = preprocess_text(document_dict)
    inverted_index = create_inverted_index(document_dict)
    pickle.dump(inverted_index, open(f"{DIRNAME}/model_queries_21CS10083.bin", "wb"))
    print("Inverted index saved in model_queries_21CS10083.bin")

if __name__ == "__main__":
    main()
                
    
    
    
