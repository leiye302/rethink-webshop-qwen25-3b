"""Read-only generation diagnostics; never gate action execution."""
import re


def protocol(text):
    raw = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    reasoning = re.search(r'<think>(.*?)</think>', raw, re.S)
    match = re.fullmatch(r'\s*<think>(.*?)</think>\s*(<RETHINK>\s*)?<action>(.*?)</action>\s*', raw, re.S)
    counts = all(raw.count(tag) == 1 for tag in ('<think>', '</think>', '<action>', '</action>'))
    strict = bool(match and match[1].strip() and match[3].strip() and counts
                  and raw.count('<RETHINK>') == int(bool(match[2])))
    return {'reasoning_open': '<think>' in raw,
            'reasoning_nonempty_closed': bool(reasoning and reasoning[1].strip()),
            'marker_present': '<RETHINK>' in raw, 'strict_protocol': strict}
