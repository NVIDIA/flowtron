import re
_ampm_re = re.compile(r'([0-9]|0[0-9]|1[0-9]|2[0-3]):?([0-5][0-9])?\s*([AaPp][Mm]\b)')


def _expand_ampm(m):
    matches = list(m.groups(0))
    txt = matches[0]
    if matches[1] == 0 or matches[1] == '0' or matches[1] == '00':
        pass
    else:
        txt += ' ' + matches[1]

    if matches[2][0] == 'a':
        txt += ' AM'
    elif matches[2][0] == 'p':
        txt += ' PM'

    return txt


def normalize_datestime(text):
    text = re.sub(_ampm_re, _expand_ampm, text)
    text = re.sub(r"([0-9]|0[0-9]|1[0-9]|2[0-3]):([0-5][0-9])?", r"\1 \2", text)
    return text
