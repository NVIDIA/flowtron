""" from https://github.com/keithito/tacotron """

import inflect
import re
_large_numbers = '(trillion|billion|million|thousand|hundred)'
_measurements = '(f|c|k|d)'
_measurements_key = {'f': 'fahrenheit', 'c': 'celsius', 'k': 'thousand', 'd': 'd'}
_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+[ ]?{}?)'.format(_large_numbers), re.IGNORECASE)
_measurement_re = re.compile(r'([0-9\.\,]*[0-9]+(\s)?{}\b)'.format(_measurements), re.IGNORECASE)
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r"[0-9]+'s|[0-9]+")

def _remove_commas(m):
  return m.group(1).replace(',', '')


def _expand_decimal_point(m):
  return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)

    # check for million, billion, etc...
    parts = match.split(' ')
    if len(parts) == 2 and len(parts[1]) > 0 and parts[1] in _large_numbers:
        return "{} {} {} ".format(parts[0], parts[1], 'dollars')

    parts = parts[0].split('.')
    if len(parts) > 2:
        return match + " dollars"    # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return "{} {}, {} {} ".format(
            _inflect.number_to_words(dollars), dollar_unit,
            _inflect.number_to_words(cents), cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return "{} {} ".format(_inflect.number_to_words(dollars), dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return "{} {} ".format(_inflect.number_to_words(cents), cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_measurement(m):
    _, number, measurement = re.split('(\d+(?:\.\d+)?)', m.group(0))
    number = _inflect.number_to_words(number)
    measurement = "".join(measurement.split())
    measurement = _measurements_key[measurement.lower()]
    return "{} {}".format(number, measurement)


def _expand_number(m):
    _, number, suffix = re.split(r"(\d+(?:'\d+)?)", m.group(0))
    num = int(number)
    if num > 1000 and num < 3000:
        if num == 2000:
            text = 'two thousand'
        elif num > 2000 and num < 2010:
            text = 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            text = _inflect.number_to_words(num // 100) + ' hundred'
        else:
            num = _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
            num = re.sub(r'-', ' ', num)
            text = num
    else:
        num = _inflect.number_to_words(num, andword='')
        num = re.sub(r'-', ' ', num)
        num = re.sub(r',', '', num)
        text = num

    if suffix == "'s" and text[-1] == 'y':
        text = text[:-1] + 'ies'

    return text


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_measurement_re, _expand_measurement, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
