"""
This module contains everything required for generating the normalized term scores and similar pubs
used by the searching mechanism. The analyzer builds a term-document matrix both by reading from the database,
and by downloading and 'reading' new pubs. The term-document matrix is written to the database, and can be recreated
later. The 'threshold' passed to the Publication.to_dict function refers to the normalized term score computed here
"""

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
    """
    This class represents a single document. Properties are provided for accessing the raw term frequency and
    normalized term scores in the document.
    """

    @property
    def euclidean_normalized(self):
        """
        Returns the Euclidean normalized term vector. That is, an n-dimensional vector (where n is the number of terms
        in the collection's vocabulary), where each term's freqency in the document is divided by the Euclidean
        distance of the vector.

        normalized = Vterms / |Vterms|, where Vterms is the terms vector, and |Vterms| = sqrt(sum of squares of the
        number of occurrences of each term)
        :return: An array of values, indexed to the collection's vocabulary
        """
        if self._euclidean_normalized is None:
            self._generate_euclidean_normalized()

        return self._euclidean_normalized

    @property
    def normalized_term_scores(self):
        """
        Returns a list of Euclidean normalized term weights, but only for terms which occur in this document
        :return: Returns is list of tuples in the form (term, weight) for each term in the document
        """
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
        """
        Returns the term frequency - inverse document frequency score for all terms in the document

        tf-idf (for a term t) = # of occurrences of T in doc * log (number of docs / number of docs where T occurrs)
        :return: A dictionary, where terms are the keys and tf-idf weights are the values
        """
        if self._tf_idf is None:
            self._generate_tf_idf()

        return self._tf_idf

    def __init__(self, filename=None):
        """
        Initalize a document. If filename is provided, opens the file and extracts word counts from the file.
        If no filename, creates an empty document
        :param filename: File to open and extract
        """
        self.filename = filename

        if filename:
            self.term_frequency = extract_word_count_from_file(filename)
        else:
            self.term_frequency = {}

        self.collection = None
        self._tf_idf = None
        self._euclidean_normalized = None
        self._normalized_term_scores = None
        self.similar_docs = None

    def __contains__(self, item):
        """
        Checks if the term occurs in the document
        :param item: The term to check for
        :return: True if term is in the document; false otherwise
        """
        return item in self.term_frequency

    def __len__(self):
        """
        The total number of terms in the document
        :return: The number of terms (non-distinct) in the document
        """
        return sum(self.term_frequency.values())

    def __hash__(self):
        """
        Hashes on the filename
        :return: returns the hash of the filename
        """
        return hash(self.filename)

    def tf_idf_score(self, term):
        """
        Returns the term freqency - inverse document freqency of the term (see tf_idf property above)
        :param term: The term
        :return: The tf-idf val (or 0 if the term is not in the document)
        """
        if term not in self.term_frequency:
            return 0
        return self.term_frequency[term] * self.collection.inverse_document_freq(term)

    def similar(self, other):
        """
        Generates a similarity value between this document and other document.
        The similarity value is the dot product of the Euclidean normalized vectors of this document and the other doc
        :param other: The other document (as a Document)
        :return: The similarity score, as a floating point number
        """
        return self.euclidean_normalized.dot(other.euclidean_normalized)

    def _generate_tf_idf(self):
        """
        Generates the tf_idf dictionary. See tf_idf property above for explenation
        """
        self._tf_idf = {term: self.tf_idf_score(term) for term in self.term_frequency}

    def _generate_euclidean_normalized(self):
        """
        Generates the euclidean normalized vector for the document. See the property above for a description
        """
        # calculate the Euclidean distance - the sqrt of the sum of squares of tf-idf values
        distance = sqrt(sum(x * x for x in self.tf_idf.values()))

        # generate a vector across the entire vocabulary
        vector = np.zeros(len(self.collection.vocabulary))

        # iterate over the vocabulary; for every term, if it is in the document, put the normalized value in the vector
        for idx, term in enumerate(self.collection.vocabulary):
            if term in self.tf_idf:
                vector[idx] = self.tf_idf[term] / distance

        self._euclidean_normalized = vector

    def top_terms(self, threshold=0.1):
        """
        Returns a list of the top weighted terms in the document, with their weights
        :param threshold: The threshold to use. Term weights above this threshold will be in the returned results
        :return: A list of (term, weight) tuples for all terms with weights above the threshold
        """
        terms = []

        for term, val in self.normalized_term_scores:
            if val < threshold:
                break
            terms.append((term, val))

        return terms


class DocumentCollection:
    """
    This class represents a collection of documents. It provides methods for determining the tf-idf values for all
    documents, and similarity scores between each pair of documents

    It is also a context manager. When in a context (with statement), add documents to the collection with add_document
    When it leaves the context, it will automatically calculate the tf_idf for all documents
    """

    @property
    def vocabulary(self):
        """
        The list of all terms occurring anywhere in the collection.
        :return: A list of terms, sorted in lexigraphic order
        """
        if not self._vocabulary:
            self._vocabulary = sorted(self.document_term_count.keys())

        return self._vocabulary

    def __init__(self):
        self.documents = []
        self.document_term_count = Counter()
        self.idf = {}
        self._vocabulary = None

    def __iter__(self):
        """
        Iterates over the documents in the collection
        :return: An iterator over the documents in the collection
        """
        return iter(self.documents)

    def add_document(self, other):
        """
        Adds a new document. Updates the document term count for all the terms in the new document
        :param other: The new Document being added
        :return: No return; but updates the document_term_count
        """
        self.documents.append(other)
        other.collection = self

        for word in other.term_frequency:
            self.document_term_count[word] += 1

        # reset the idf and vocabulary
        self.idf = {}
        self._vocabulary = None

    def inverse_document_freq(self, term):
        """
        Determines the inverse document frequency of a term.
        idf = log (number of documents / number of documents which contain the term)
        :param term: The specified term
        :return: The idf of the term
        """
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
        """
        When exiting a context, updates the idf for all terms in the vocabulary
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        num_docs = len(self.documents)
        self.idf = {term: log(num_docs / occurrences) for (term, occurrences) in self.document_term_count.items()}

    def determine_similar(self, threshold=0.5):
        """
        Determines document similarities between each pair of documents
        :param threshold: The threshold for similarity. Documents are 'similar' if their similarity score is above
        the threshold
        :return: A dictionary, keyed on Documents, where each value is a list of similar Documents
        """
        similar_docs_dict = {}

        for doc in self.documents:
            similar_docs = []
            for other_doc in (d for d in self.documents if d is not doc):
                if doc.similar(other_doc) >= threshold:
                    similar_docs.append(other_doc)

            doc.similar_docs = similar_docs
            similar_docs_dict[doc] = similar_docs


def process_response(response):
    """
    Takes in a response object, saves the content and creates a new Document from it
    :param response: A response object, containing a downloaded PDF
    :return: A new Document object, from the response's content; also writes the PDF to file
    """
    if not response or response.status_code >= 400 or not response.content:
        return None

    filename = response.url.split('/')[-1]
    print('Saving file '.format(filename))

    with open(filename, 'wb') as file:
        file.write(response.content)

    return Document(filename)


def process_document(filename):
    """
    Attempts to create a Document from a file
    :param filename: the filename of the PDF file
    :return: A Document, or None if an exception occurs
    """
    try:
        print("Analyzing {}...".format(filename))
        return Document(filename)
    except:
        print("UNABLE TO PROCESS FILE ", filename)
        return None


def queue_downloads(pubs):
    """
    Creates and yields Futures for downloading every pub listed in pubs
    :param pubs: A list of Publications to download
    :returns: yields future objects
    """
    responses = []
    with futures.ThreadPoolExecutor(max_workers=20) as executor:
        for pub in pubs:
            print("Downloading pub {}".format(pub.publication_url))
            future = executor.submit(requests.get, pub.publication_url)
            responses.append(future)

    for future in responses:
        yield future.result()


def queue_processing_responses(responses):
    """
    Takes an iterator of response Futures and yields an iterator of pending Document futures
    :param responses: An iterator produced from queue_downloads
    :return: An iterator of document Futures
    """
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
    """
    Like queue_processing_responses, but with local files instead
    :param pubs: A list of pubs to load from file
    :return: An iterator over document futures
    """
    with futures.ProcessPoolExecutor() as executor:
        for pub in pubs:
            future = executor.submit(process_document, pub.publication_url.split('/')[-1])
            yield future


def load_documemts(pubs):
    """
    Either retrieves pubs from the internet, or loads them from file. Returns an iterator of Documents
    :param pubs: A list of Publications
    :return: an iterator over Documents
    """
    print("Loading new pubs from the Internet or disk...")
    downloads = []
    dir_list = os.listdir()
    with futures.ThreadPoolExecutor() as executor:
        for pub in pubs:
            filename = pub.publication_url.split('/')[-1]
            if filename in dir_list:
                print("Loading {} from disk".format(filename))
                downloads.append(None)
            else:
                print("Downloading {}".format(pub.publication_url))
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
    """
    Loads the term freqency dictionary for a publication from the database
    :param pub: The Publication to load
    :return: A Document object, with the term-freqency dictionary populated
    """
    print("Loading {} from database...".format(pub.title))
    doc = Document()
    doc.filename = pub.publication_url.split('/')[-1]

    for association in pub.terms:
        doc.term_frequency[association.term.term] = association.occurs

    return doc


def load_docs_from_database(pubs):
    """
    Loads a list of Publications from the database
    :param pubs: A list of Publications to load into Document objects
    :return: An iterator over new Document objects
    """
    # can't thread this due to SQLite lock
    print('Loading from database...')
    for pub in pubs:
        yield load_doc_from_database(pub)


def run_analyzer(new_pubs=None, session=None, similar_threshold=0.15, term_threshold=0.025):
    """
    Entry point into this module. Builds the document-term matrix (either loads from the database, or builds from
    analyzing the files. Determines term weights for each document, and document similarities, then writes to the
    database.
    :param term_threshold: the threshold in use for terms
    :param similar_threshold: the similarity threshold
    :param new_pubs: a list of new Publications, which will be downloaded if they are not found locally
    :param session: The database session
    :return: None; but updates the session (does NOT commit changes)
    """
    docs = DocumentCollection()

    pubs = session.query(Publication).all()

    if not new_pubs:
        new_pubs = []

    # these are the existing Publications, which are already in the database and have already been analyzed
    pubs_to_load = [p for p in pubs if p not in new_pubs]
    loaded_docs = load_docs_from_database(pubs_to_load)

    # adds all the documents to the DocumentCollection. The context manager ensures that the idf is automatically
    # generated at the end of the with block
    docs_to_pubs = {}
    with docs:
        for pub, doc in zip(new_pubs, load_documemts(new_pubs)):
            if doc:
                print('Adding document '.format(doc.filename))
                # os.remove(doc.filename)
                docs.add_document(doc)
                docs_to_pubs[doc] = pub
        for pub, doc in zip(pubs_to_load, loaded_docs):
            docs.add_document(doc)
            docs_to_pubs[doc] = pub
            print('Adding DB loaded document'.format(doc.filename))

    docs.determine_similar(similar_threshold)
    print("Total terms in vocabulary: ", len(docs.vocabulary))

    # ensure that all terms are in the database, and build a dictionary where the key is the term (as a string),
    # and the value is the database row object
    print("Building term to DB object linkage...")
    term_to_obj = {}
    for term in docs.vocabulary:
        term_obj = session.query(SearchTerm).filter_by(term=term).one_or_none()

        if not term_obj:
            term_obj = SearchTerm(term=term)
            session.add(term_obj)

        term_to_obj[term] = term_obj

    top_terms = set()
    # Iterate over the documents to write the results to the database (the term-freqencies and term weights)
    # This updates term-weights for existing documents
    for doc in docs:
        pub = docs_to_pubs[doc]
        print("Doc filename: ", doc.filename)
        print("Pub title: ", pub.title)
        print("Total terms in doc: ", len(doc.term_frequency))

        doc_terms = doc.top_terms(term_threshold)

        print("Number of top terms (threshold {}): {}".format(term_threshold, len(doc_terms)))
        pprint(doc_terms)

        top_terms.update(set(t for (t, v) in doc_terms))

        print("similar pubs:")
        for similar in doc.similar_docs:
            similar_pub = docs_to_pubs[similar]

            if similar_pub and similar_pub not in pub.similar:
                print(similar_pub.title)
                pub.similar.append(similar_pub)

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
            association.weight = doc_term_weights[
                association.term.term] if association.term.term in doc_term_weights else 0

    print("Total top terms: ", len(top_terms))
