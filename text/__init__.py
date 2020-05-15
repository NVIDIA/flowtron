""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import symbols
from text.symbols import _punctuation as punctuation_symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# for arpabet with apostrophe
_apostrophe = re.compile(r"(?=\S*['])([a-zA-Z'-]+)")

def text_to_sequence(text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

        The text can optionally have ARPAbet sequences enclosed in curly braces embedded
        in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

        Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through

        Returns:
            List of integers corresponding to the symbols in the text
    '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(text)
            break
        sequence += _symbols_to_sequence(m.group(1))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)

    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


def get_arpabet(word, cmudict, index=0):
    re_start_punc = r"\A\W+"
    re_end_punc = r"\W+\Z"

    start_symbols = re.findall(re_start_punc, word)
    if len(start_symbols):
        start_symbols = start_symbols[0]
        word = word[len(start_symbols):]
    else:
        start_symbols = ''

    end_symbols = re.findall(re_end_punc, word)
    if len(end_symbols):
        end_symbols = end_symbols[0]
        word = word[:-len(end_symbols)]
    else:
        end_symbols = ''

    arpabet_suffix = ''
    if _apostrophe.match(word) is not None and word.lower() != "it's" and word.lower()[-1] == 's':
        word = word[:-2]
        arpabet_suffix = ' Z'
    arpabet = None if word.lower() in HETERONYMS else cmudict.lookup(word)

    if arpabet is not None:
        return start_symbols + '{%s}' % (arpabet[index] + arpabet_suffix) + end_symbols
    else:
        return start_symbols + word + end_symbols


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

HETERONYMS = set(files_to_list('data/heteronyms'))
