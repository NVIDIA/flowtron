""" adapted from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from .numbers import normalize_numbers
from .acronyms import normalize_acronyms
from .datestime import normalize_datestime


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('ms', 'miss'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

_safe_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('no', 'number'),
]]



def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_safe_abbreviations(text):
    for regex, replacement in _safe_abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_numbers(text):
    return normalize_numbers(text)


def expand_acronyms(text):
    return normalize_acronyms(text)


def expand_datestime(text):
    return normalize_datestime(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def separate_acronyms(text):
    text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)
    return text


def remove_hyphens(text):
    text = re.sub(r'(?<=\w)(-)(?=\w)', ' ', text)
    return text


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def flowtron_cleaners(text):
    text = collapse_whitespace(text)
    text = remove_hyphens(text)
    text = expand_datestime(text)
    text = expand_numbers(text)
    text = expand_safe_abbreviations(text)
    text = expand_acronyms(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, with number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text
