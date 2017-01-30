'''
This module is responsible for loading a PDF file (by filename) and returning a word count of all words in the file
'''

from io import StringIO
from collections import Counter
import logging

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfparser import PDFDocument, PDFParser, logger as PDFParserLogger
from pdfminer.layout import LAParams

from PyPDF2 import PdfFileReader

PDFParserLogger.setLevel(logging.ERROR)

def extract_word_count_from_file_pdfminer(filename):
    """
    Uses PdfMiner to extract text, then uses a counter to count occurrences
    :param filename: The filename to read
    :return: A Counter of term - occurrences
    """
    resources = PDFResourceManager()
    string_buffer = StringIO()
    layout = LAParams()
    device = TextConverter(resources, string_buffer, laparams=layout)

    file = open(filename, 'rb')
    interpreter = PDFPageInterpreter(resources, device)
    parser = PDFParser(file)

    document = PDFDocument()
    document.initialize()

    parser.set_document(document)
    document.set_parser(parser)

    for page in document.get_pages():
        interpreter.process_page(page)

    file.close()

    string_buffer.seek(0)
    return determine_counts(string_buffer.getvalue())


def extract_word_count_from_file_pypdf2(filename):
    """
    Uses PyPDF2 to extract text
    :param filename: The file to read
    :return: A Counter of term - occurrences
    """
    with open(filename, 'rb') as file:
        pdf_doc = PdfFileReader(file, strict=False)

        word_counts = Counter()

        for page in pdf_doc.pages:
            try:
                word_counts += determine_counts(page.extractText())
            except:
                pass

        return word_counts


def extract_word_count_from_file(filename):
    """
    Extracts word counts from a file. First attempts pdfminer. If it doesn't work, tries pypdf2.
    :param filename: The file to read
    :return: A Counter of term - occurrences, or None if neither library could read the file
    """
    try:
        print("Processing ", filename)
        return extract_word_count_from_file_pdfminer(filename)
    except:
        print("PDFMiner unable to process {}. Switching to PyPDF2...".format(filename))
        try:
            return extract_word_count_from_file_pypdf2(filename)
        except:
            print("Unable to extract text from file ", filename)
            return None


def determine_counts(text):
    """
    Determines the term counts of a piece of text
    :param text: The text to analyze
    :return: A Counter of term - occurrences
    """
    word_count = Counter()

    processed_text = remove_unwanted_char(text)
    # split on the sentences
    for sentence in processed_text.split('.'):
        words = [w for w in sentence.split() if test_valid(w)]

        word_count.update(words)

    return word_count


def remove_unwanted_char(text, unwanted='()"', puncuation='!?;:'):
    """
    Removes unwanted characters and puncuation
    :param text: Text to process
    :param unwanted: Characters to remove
    :param puncuation: Characters to turn into periods (.)
    :return: A string with unwanted characters removed and puncuation turned into .
    """
    new_text = text.lower().replace('\n', ' ')
    new_text = new_text.replace('- ', ' ')

    for ch in unwanted:
        new_text = new_text.replace(ch, ' ')
    for ch in puncuation:
        new_text = new_text.replace(ch, '.')

    return new_text


def test_valid(word):
    """
    Tests if a term is valid. A valid term is at least 2, but less than 20 characters, containing only alphanumeric
    characters (but not all numbers - that is a number not a term).
    :param word: The term to check
    :return: True if the term is valid, false if not
    """
    if len(word) < 2 or len(word) > 20:
        return False

    if word.isalnum() and not (word.isdecimal() or word.isnumeric() or word.isdigit()):
        return True

    return False
