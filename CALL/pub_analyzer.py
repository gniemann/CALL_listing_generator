from collections import Counter
from concurrent import futures
from math import log, sqrt
from pprint import pprint
import os

import numpy as np
import requests

from CALL.pdf_parser import extract_word_count_from_file
from CALL.models import Publication, SearchTerm, SearchTerms


class Document:
    @property
    def euclidean_normalized(self):
        if self._euclidean_normalized is None:
            self._generate_euclidean_normalized()

        return self._euclidean_normalized

    @property
    def normalized_term_scores(self):
        if not self._normalized_term_scores:
            scores = []
            for (term, val) in zip(self.collection.vocabulary, self.euclidean_normalized):
                if val > 0:
                    scores.append((term, val))

            scores.sort(key=lambda x: x[1], reverse=True)
            self._normalized_term_scores = scores

        return self._normalized_term_scores

    @property
    def tf_idf(self):
        if self._tf_idf is None:
            self._generate_tf_idf()

        return self._tf_idf

    def __init__(self, filename=None):
        self.filename = filename

        if filename:
            self.term_frequency = extract_word_count_from_file(filename)
        else:
            self.term_frequency = {}

        self.collection = None
        self._tf_idf = None
        self._euclidean_normalized = None
        self._normalized_term_scores = None
        self.similiar_docs = None

    def __contains__(self, item):
        return item in self.term_frequency

    def __len__(self):
        return sum(self.term_frequency.values())

    def __hash__(self):
        return hash(self.filename)

    def tf_idf_score(self, term):
        if term not in self.term_frequency:
            return 0
        return self.term_frequency[term] * self.collection.inverse_document_freq(term)

    def similiar(self, other):
        return self.euclidean_normalized.dot(other.euclidean_normalized)

    def _generate_tf_idf(self):
        self._tf_idf = {term: self.tf_idf_score(term) for term in self.term_frequency}


    def _generate_euclidean_normalized(self):
        # calculate the Euclidean distance - the sqrt of the sum of squares of tf-idf values
        distance = sqrt(sum(x * x for x in self.tf_idf.values()))

        # generate a vector across the entire vocabulary
        vector = np.zeros(len(self.collection.vocabulary))

        #iterate over the vocabulary; for every term, if it is in the document, put the normalized value in the vector
        for idx, term in enumerate(self.collection.vocabulary):
            if term in self.tf_idf:
                vector[idx] = self.tf_idf[term] / distance

        self._euclidean_normalized = vector

    def top_terms(self, threshold=0.1):
        terms = []

        for term, val in self.normalized_term_scores:
            if val < threshold:
                break
            terms.append((term, val))

        return terms

class DocumentCollection:
    @property
    def vocabulary(self):
        if not self._vocabulary:
            self._vocabulary = sorted(self.document_term_count.keys())

        return self._vocabulary

    def __init__(self):
        self.documents = []
        self.document_term_count = Counter()
        self.idf = {}
        self._vocabulary = None

    def __iter__(self):
        return iter(self.documents)

    def add_document(self, other):
        self.documents.append(other)
        other.collection = self

        for word in other.term_frequency:
            self.document_term_count[word] += 1

        # reset the idf and vocabulary
        self.idf = {}
        self._vocabulary = None

    def inverse_document_freq(self, term):
        if term not in self.idf:
            doc_term_cnt = self.document_term_count[term]
            val = 0
            if doc_term_cnt > 0:
                val = log(len(self.documents) / doc_term_cnt)

            self.idf[term] = val

        return self.idf[term]

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        numDocs = len(self.documents)
        self.idf = {term: log(numDocs / occurrences) for (term, occurrences) in self.document_term_count.items()}

    def determine_similiar(self, threshold=0.5):
        similiar_docs_dict = {}

        for doc in self.documents:
            similiar_docs = []
            for other_doc in (d for d in self.documents if d is not doc):
                if doc.similiar(other_doc) >= threshold:
                    similiar_docs.append(other_doc)

            doc.similiar_docs = similiar_docs
            similiar_docs_dict[doc] = similiar_docs

def process_response(response):
    if not response or response.status_code >= 400 or not response.content:
        return None

    filename = response.url.split('/')[-1]
    print('Saving file '.format(filename))

    with open(filename, 'wb') as file:
        file.write(response.content)

    return Document(filename)

def process_document(filename):
    try:
        return Document(filename)
    except:
        print("UNABLE TO PROCESS FILE ", filename)
        return None

def queue_downloads(pubs):
    responses = []
    with futures.ThreadPoolExecutor(max_workers=20) as executor:
        for pub in pubs:
            print("Downloading pub {}".format(pub.publication_url))
            future = executor.submit(requests.get, pub.publication_url)
            responses.append(future)

    for future in responses:
        yield future

def queue_processing_responses(responses):
    new_docs = []
    with futures.ProcessPoolExecutor() as executor:
        for future in responses:
            res = future.result()
            if res:
                next_future = executor.submit(process_response, res)
                new_docs.append(next_future)

    for future in new_docs:
        yield future

def queue_load_files(pubs):
    new_docs = []
    with futures.ProcessPoolExecutor() as executor:
        for pub in pubs:
            future = executor.submit(process_document, pub.publication_url.split('/')[-1])
            yield future

def load_documemts(pubs):
    downloads = []
    dir_list = os.listdir()
    with futures.ThreadPoolExecutor() as executor:
        for pub in pubs:
            filename = pub.publication_url.split('/')[-1]
            if filename in dir_list:
                downloads.append(None)
            else:
                future = executor.submit(requests.get, pub.publication_url)
                downloads.append(future)

    doc_futures = []
    with futures.ProcessPoolExecutor() as executor:
        for future, pub in zip(downloads, pubs):
            if not future:
                doc_future = executor.submit(process_document, pub.publication_url.split('/')[-1])
                doc_futures.append(doc_future)
            else:
                response = future.result()
                doc_future = executor.submit(process_response, response)
                doc_futures.append(doc_future)

    for future in doc_futures:
        yield future.result()

def load_doc_from_database(pub):
    print("Loading {} from database...".format(pub.title))
    doc = Document()
    doc.filename = pub.publication_url.split('/')[-1]

    for association in pub.terms:
        doc.term_frequency[association.term.term] = association.occurs

    return doc

def load_docs_from_database(pubs):
    print('Loading from database...')
    pending = []

    with futures.ThreadPoolExecutor as executor:
        for pub in pubs:
            future = executor.submit(load_doc_from_database, pub)
            pending.append(future)

    for future in pending:
        yield future


SAVE_TO_DATABASE = True

SIMILIAR_THRESHOLD = .15
TOP_TERMS_THRESHOLD = .025

def run_analyzer(new_pubs=None, session=None):
    docs = DocumentCollection()

    pubs = session.query(Publication).all()

    if not new_pubs:
        new_pubs = []

    pubs_to_load = [p for p in pubs if p not in new_pubs]
    loaded_docs = load_docs_from_database(pubs_to_load)

    docs_to_pubs = {}
    with docs:
        for pub, doc in zip(new_pubs, load_documemts(new_pubs)):
            if doc:
                print('Adding document '.format(doc.filename))
                #os.remove(doc.filename)
                docs.add_document(doc)
                docs_to_pubs[doc] = pub
        for pub, doc in zip(pubs_to_load, loaded_docs):
            docs.add_document(doc)
            docs_to_pubs[doc] = pub
            print('Adding DB loaded document'.format(doc.filename))

    docs.determine_similiar(SIMILIAR_THRESHOLD)
    print("Total terms in vocabulary: ", len(docs.vocabulary))

    print("Building term to DB object linkage...")
    term_to_obj = {}
    for term in docs.vocabulary:
        term_obj = session.query(SearchTerm).filter_by(term=term).one_or_none()

        if not term_obj:
            term_obj = SearchTerm(term=term)
            session.add(term_obj)

        term_to_obj[term] = term_obj

    top_terms = set()
    for doc in docs:
        pub = docs_to_pubs[doc]
        print("Doc filename: ", doc.filename)
        print("Pub title: ", pub.title)
        print("Total terms in doc: ", len(doc.term_frequency))

        doc_terms = doc.top_terms(TOP_TERMS_THRESHOLD)

        print("Number of top terms (threshold {}: {}".format(TOP_TERMS_THRESHOLD, len(doc_terms)))
        pprint(doc_terms)

        top_terms.update(set(t for (t,v) in doc_terms))

        print("Similiar pubs:")
        for similiar in doc.similiar_docs:
            similiar_pub = docs_to_pubs[similiar]

            if similiar_pub and similiar_pub not in pub.similiar:
                print(similiar_pub.title)
                pub.similiar.append(similiar_pub)


        # set the search terms - first get the term (or create it), than set the linkage
        # first get the list of terms
        doc_terms = [t.term for t in pub.terms]
        doc_term_weights = {term: val for (term, val) in doc.normalized_term_scores}
        for term, val in doc.term_frequency.items():
            term_obj = term_to_obj[term]

            if term_obj not in doc_terms:
                association = SearchTerms(occurs=val)
                association.term = term_obj
                pub.terms.append(association)

        for association in pub.terms:
            association.weight = doc_term_weights[association.term.term] if association.term.term in doc_term_weights else 0



    print("Total top terms: ", len(top_terms))


if __name__ == '__main__':
    run_analyzer()





