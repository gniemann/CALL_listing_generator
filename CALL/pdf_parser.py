'''
This module is responsible for loading a PDF file (by filename) and returning a word count of all words in the file
'''


from io import StringIO
from collections import Counter

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfparser import PDFDocument, PDFParser
from pdfminer.layout import LAParams

from PyPDF2 import PdfFileReader

def extract_word_count_from_file_pdfminer(filename):
    resources = PDFResourceManager()
    string_buffer = StringIO()
    codec = 'utf-8'
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
    with open(filename, 'rb') as file:
        pdf_doc = PdfFileReader(file, strict=False)

        word_counts = Counter()

        for page in pdf_doc.pages:
            try:
                word_counts = word_counts + determine_counts(page.extractText())
            except:
                pass

        return word_counts

def extract_word_count_from_file(filename):
    try:
        return extract_word_count_from_file_pdfminer(filename)
    except:
        print("PDFMiner unable to process {}. Switching to PyPDF2...".format(filename))
        try:
            return extract_word_count_from_file_pypdf2(filename)
        except:
            print("Unable to extract text from file ", filename)
            return None

def determine_counts(text):
    word_count = Counter()
    word_pair_count = Counter()

    processed_text = remove_unwanted_char(text)
    # split on the sentences
    for sentence in processed_text.split('.'):
        words = [w for w in sentence.split() if test_valid(w)]

        word_count.update(words)


    return word_count

def remove_unwanted_char(text, unwanted='()"', puncuation='!?;:'):
    new_text = text.lower().replace('\n', ' ')
    nex_text = new_text.replace('- ', '')

    for ch in unwanted:
        new_text = new_text.replace(ch, ' ')
    for ch in puncuation:
        new_text = new_text.replace(ch, '.')

    return new_text

def test_valid(word):
    if len(word) < 2 or len(word) > 20:
        return False

    if word.isalnum() and not (word.isdecimal() or word.isnumeric() or word.isdigit()):
        return True

    return False