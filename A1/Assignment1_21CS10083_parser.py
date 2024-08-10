import os
import re
import spacy
import nltk
import sys
DIRNAME = os.path.dirname(__file__)

nltk.download('wordnet')    # download the wordnet corpus for lemmatization (only once)
nltk.download('stopwords')  # download the stopwords corpus for removing stopwords (only once)

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
                documents_dict[int(text[i][1].strip())] = text[i+1][1].strip()
                while j < len(text) and text[j][0] != '.I':
                    documents_dict[int(text[i][1].strip())] += text[j-1][1].strip()
                    j += 1
                i = j - 1
            i += 1
    return documents_dict
            
def preprocess_text(query_dict):
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
    for key in query_dict:
        query_dict[key] = re.sub(r'[^\w\s]|\\n', ' ', query_dict[key])
        query_dict[key] = nlp(query_dict[key].lower())
        query_dict[key] = [stemmer.stem(word.lemma_) for word in query_dict[key] if not word.is_stop]
        query_dict[key] = [word for word in query_dict[key] if word.strip() not in [' ', '', '\n', '\t']]
    return query_dict
def save_txt_file(query_dict):
    """
    Saves the preprocessed text in a txt file
    :param query_dict: the dictionary of id and preprocessed text
    """
    with open(f"{DIRNAME}/queries_21CS10083.txt", "w") as f:
        for key in query_dict:
            f.write(f"{key}\t{' '.join(query_dict[key])}\n")
    print("Queries saved in queries.txt")

def main():
    QUERY_FILE = sys.argv[1]
    query_dict = extract_id_and_text(QUERY_FILE)
    query_dict = preprocess_text(query_dict)
    save_txt_file(query_dict)

if __name__ == "__main__":
    main()
                
    
    
    