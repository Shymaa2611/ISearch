from django.shortcuts import render
import glob
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import string
nltk.download('punkt')
nltk.download("stopwords")
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download("wordnet")
import numpy as np
import math
from .forms import SearchForm



docs_paths = glob.glob(os.path.join("C:\\Users\\dell\\Desktop\\IR_Project\\wikIR1k\\documents", '*.txt'))
paths=[]
count=0
for path in docs_paths:
    paths.append({"doc_id":count,"path":path})
    count+=1

def search_engine(request):
    relevant=None
    if request.method=='POST':
        form=SearchForm(request.POST)
        if form.is_valid():
            form.save()
            query=form.cleaned_data['query']
            model=form.cleaned_data['model']
            if model=="Document Term Matrix OR Bitwise":
                relevant=document_term_matrix_OR(query)
            elif model=="Document Term Matrix AND Bitwise":
                relevant=document_term_matrix_AND(query)
            elif model=="Inverted Index OR":
                relevant=inverted_index_OR(query)
                all_ids=[]
                for rel in relevant:
                    rele=""
                    val=paths[rel]['path'][51:57]
                    for re in val:
                        if re=='.':
                            break
                        else:
                            rele+=re
                    all_ids.append(rele)
                relevant=list(set(all_ids))
            elif model=="Inverted Index AND":
                relevant=inverted_index_AND(query)
                all_ids=[]
                for rel in relevant:
                    rele=""
                    val=paths[rel]['path'][51:57]
                    for re in val:
                        if re=='.':
                            break
                        else:
                            rele+=re
                    all_ids.append(rele)
                relevant=list(set(all_ids))

            else:
                relevant=tfidf(query)
     
    else:
        form=SearchForm()
    context={
        'form':form,
        'Relevance':relevant

    }

    return render(request,'search_engine.html',context)

def read_dataset():
    docs_paths = glob.glob(os.path.join(f"C:\\Users\\dell\\Desktop\\IR_Project\\wikIR1k\\documents", '*.txt'))
    all_docs=[]
    for docs in docs_paths:
       with open(docs, 'r') as file:
          lines = file.read()
          all_docs.append(lines)
    docs=[]
    for doc in all_docs:
       text=clean(doc)
       docs.append(text)
    return docs

def remove_html_tags(text):
    html_pattern = r'<.*?>'
    without_html = re.sub(pattern=html_pattern, repl=' ', string=text)
    return without_html

def convert_to_lower(text):
    return text.lower()


def remove_urls(text):
    url_pattern = r'https?://\S+|www\.\S+'
    without_urls = re.sub(pattern=url_pattern, repl=' ', string=text)
    return without_urls


def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number


def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc


def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)


def clean(text):
    text=remove_html_tags(text)
    text=convert_to_lower(text)
    text=remove_urls(text)
    text=remove_numbers(text)
    text=remove_punctuation(text)
    text=remove_extra_white_spaces(text)
    text=remove_stopwords(text)
    text=lemmatizing(text)
    return text

def read_query(query):
    query=clean(query)
    query_terms=query.split()
    return query_terms

def document_term_matrix_AND(query):
    docs=read_dataset()
    query_terms=read_query(query)
    unique_terms = {term for doc in docs for term in doc.split()}
    doc_term_matrix = {}
    for term in unique_terms:
        doc_term_matrix[term] = []

        for doc in docs:
           if term in doc:
              doc_term_matrix[term].append(1)
           else: doc_term_matrix[term].append(0)
    relevant_docs = []
    count=0
    v1 = np.array(doc_term_matrix[query_terms[0]])
    for term in range(1,len(query_terms)-1):
        v2 = np.array(doc_term_matrix[query_terms[term]])
        v1 = v1 & v2
    for v in v1:
        if v==1:
         # document_model_instance = documentModel(document=relevant_docs[count])
         # document_model_instance.save()
          relevant_docs.append(f"{paths[count]['path'][51:57]}")
        count += 1

    relevant_docs = list(set(relevant_docs))
    new_relevant_docs=[]
    count=0
    for relevant in  relevant_docs:
        relevant_doc=""
        for re in relevant:
            if re=='.':
               break
            else:
                relevant_doc+=re
        new_relevant_docs.append(relevant_doc)

    #for doc in relevant_docs:
    #     document_model_instance = documentModel(document=doc)
    #     document_model_instance.save()
    return new_relevant_docs

def document_term_matrix_OR(query):
    docs=read_dataset()
    query_terms=read_query(query)
    unique_terms = {term for doc in docs for term in doc.split()}
    doc_term_matrix = {}
    for term in unique_terms:
        doc_term_matrix[term] = []

        for doc in docs:
           if term in doc:
              doc_term_matrix[term].append(1)
           else: doc_term_matrix[term].append(0)
    relevant_docs = []
    count=0
    v1 = np.array(doc_term_matrix[query_terms[0]])
    for term in range(1,len(query_terms)-1):
        v2 = np.array(doc_term_matrix[query_terms[term]])
        v1 = v1 | v2
    for v in v1:
        if v==1:
          relevant_docs.append(f"{paths[count]['path'][51:57]}")
        count += 1

    relevant_docs = list(set(relevant_docs))
    new_relevant_docs=[]
    count=0
    for relevant in  relevant_docs:
        relevant_doc=""
        for re in relevant:
            if re=='.':
               break
            else:
                relevant_doc+=re
        new_relevant_docs.append(relevant_doc)

    #for doc in relevant_docs:
    #     document_model_instance = documentModel(document=doc)
    #     document_model_instance.save()
    return new_relevant_docs

def or_postings(posting1, posting2):
    p1 = 0
    p2 = 0
    result = list()
    while p1 < len(posting1) and p2 < len(posting2):
        if posting1[p1] == posting2[p2]:
            result.append(posting1[p1]) 
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            result.append(posting2[p2])
            p2 += 1
        else:
            result.append(posting1[p1])
            p1 += 1
    while p1 < len(posting1):
        result.append(posting1[p1])
        p1 += 1
    while p2 < len(posting2):
        result.append(posting2[p2])
        p2 += 1
    return result

def inverted_index_OR(query):
    docs=read_dataset()
    query_terms=read_query(query)
    inverted_index = {}
    for i, doc in enumerate(docs):
      for term in doc.split():
         if term in inverted_index:
            inverted_index[term].add(i)
         else: inverted_index[term] = {i}
    pl_1 = list(inverted_index[query_terms[0]])
    for term in range(1,len(query_terms)-1):
        pl_2 = list(inverted_index[query_terms[term]])
        pl_1=or_postings(pl_1, pl_2)
    return pl_1

def and_postings(posting1, posting2):
    p1 = 0
    p2 = 0
    result = list()
    while p1 < len(posting1) and p2 < len(posting2):
        if posting1[p1] == posting2[p2]:
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            p2 += 1
        else:
            p1 += 1
    return result

def inverted_index_AND(query):
    docs=read_dataset()
    query_terms=read_query(query)
    inverted_index = {}
    for i, doc in enumerate(docs):
      for term in doc.split():
         if term in inverted_index:
            inverted_index[term].add(i)
         else: inverted_index[term] = {i}
    pl_1 = list(inverted_index[query_terms[0]])
    for term in range(1,len(query_terms)-1):
        pl_2 = list(inverted_index[query_terms[term]])
        pl_1=and_postings(pl_1, pl_2)
    return pl_1

def calculateTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def calculateIDF(documents):
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        if val > 0:
            idfDict[word] = math.log(N / float(val))
    return idfDict

def calculateTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def cosine_similarity(query_tfidf, doc_tfidf_list):
    similarities = []
    for doc_tfidf in doc_tfidf_list:
        dot_product = sum(query_tfidf[token] * doc_tfidf.get(token, 0) for token in query_tfidf)
        query_norm = math.sqrt(sum(value**2 for value in query_tfidf.values()))
        doc_norm = math.sqrt(sum(value**2 for value in doc_tfidf.values()))
        similarity = dot_product / (query_norm * doc_norm) if query_norm != 0 and doc_norm != 0 else 0
        similarities.append(similarity)
    return similarities


def tfidf(query):
    docs = read_dataset()
    query_words = read_query(query)
    unique_terms = set()
    for doc in docs:
        unique_terms.update(doc.split())
    bag_of_words_docs = [doc.split() for doc in docs]
    num_of_words_docs = []
    for bag in bag_of_words_docs:
        word_count = {term: 0 for term in unique_terms}
        for word in bag:
            if word in word_count:
                word_count[word] += 1
        num_of_words_docs.append(word_count)
    tf_docs = [calculateTF(word_count, doc) for word_count, doc in zip(num_of_words_docs, bag_of_words_docs)]
    idfs = calculateIDF(num_of_words_docs)
    tfidf_docs = [calculateTFIDF(tf_doc, idfs) for tf_doc in tf_docs]
    num_of_words_query = {term: 0 for term in unique_terms}
    for word in query_words:
        if word in num_of_words_query:
            num_of_words_query[word] += 1
    tf_query = calculateTF(num_of_words_query, query_words)
    tfidf_query = calculateTFIDF(tf_query, idfs)
    similarities = cosine_similarity(tfidf_query, tfidf_docs)
    
    similarity_values = [{'Document_ID': i, 'similarity': math.floor(similarity * 100)} for i, similarity in enumerate(similarities)]
    similarity_values.sort(key=lambda x: x['similarity'], reverse=True)
    similarity_documents = [
        f"Document {entry['Document_ID']} - Similarity = {entry['similarity']}%" 
        for entry in similarity_values[:10] if entry['similarity'] > 20
    ]
    
    return similarity_documents






