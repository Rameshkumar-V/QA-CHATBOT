"""Constrained local answers for unambiguous missing-Aadhaar questions."""
from dataclasses import dataclass, field
import re

import pandas as pd

from services.data_tools import missing_mask


AADHAAR_SPELLINGS = frozenset({'aadhaar', 'aadhar', 'aadhar', 'adhaar', 'adhar', 'adaar', 'aadaar'})
QUERY_WORDS = frozenset('which student students does doesnt doesn do dont don not have has hasnt hasn '
    'an a their there any is are the give given provide provided submitted submit number numbers no '
    'id aadhaar missing blank empty list all find show me who names name of count how many without with haven '
    'havent t did didnt didn yet and roll please can you tell what details card records record'.split())
IDENTIFIER_NAMES = frozenset({'name', 'studentname', 'studentsname', 'nameofstudent', 'nameofthestudent',
    'studentfullname', 'fullname', 'candidatename',
    'roll', 'rollno', 'rollnumber', 'regno', 'registerno', 'registernumber', 'registrationno', 'registrationnumber', 'admissionnumber',
    'admissionno', 'enrollmentnumber', 'enrolmentnumber', 'studentid', 'usn'})


@dataclass
class DirectAnswer:
    answer: str
    results: dict[str, pd.DataFrame] = field(default_factory=dict)


def normalize(text):
    words = re.findall(r'\w+', str(text).casefold())
    return ' '.join('aadhaar' if word in AADHAAR_SPELLINGS else word for word in words)


def display_value(value):
    text = re.sub(r'\s+', ' ', str(value))
    text = text[:80] + ('…' if len(text) > 80 else '')
    return re.sub(r'([\\`*_{}\[\]<>|])', r'\\\1', text)


def answer_missing_aadhaar(question, tables):
    """Return None for other questions or extra conditions; never silently ignore filters."""
    normalized = normalize(question)
    words = set(normalized.split())
    absent = bool(words & {'missing', 'blank', 'empty', 'without'}) or bool(
        words & {'not', 'doesnt', 'dont', 'hasnt', 'havent', 'didnt', 'doesn', 'don', 'hasn', 'haven', 'didn'})
    if 'aadhaar' not in words or not absent:
        return None
    # Negated missing or questions about validity need a different query plan.
    if re.search(r'\b(?:not|no|without)\s+(?:missing|blank|empty|without)\b', normalized):
        return None
    scope_question = normalized
    explicit_tables = [name for name in tables if name.casefold() in question.casefold()]
    if not explicit_tables:
        explicit_tables = [name for name in tables if name.split(' :: ')[0].casefold() in question.casefold()]
    for name in explicit_tables:
        filename = name.split(' :: ')[0]
        scope_question = scope_question.replace(normalize(name), '').replace(normalize(filename), '')
    if explicit_tables:
        scope_question = re.sub(r'\b(?:in|from|for|sheet|table|dataset|file)\b', '', scope_question)
    if set(scope_question.split()) - QUERY_WORDS:
        return None
    selected = {}
    for table, frame in tables.items():
        if explicit_tables and table not in explicit_tables:
            continue
        candidates = [col for col in frame.columns if 'aadhaar' in normalize(col).split()]
        exact = [col for col in candidates if f' {normalize(col)} ' in f' {normalized} ']
        candidates = exact or candidates
        if len(candidates) > 1:
            return DirectAnswer(f'Multiple Aadhaar columns were found in {display_value(table)}: '
                                + ', '.join(display_value(col) for col in candidates)
                                + '. Please name the column to check.')
        if candidates:
            selected[table] = candidates[0]
    if not selected:
        return DirectAnswer('I could not identify an Aadhaar column in the uploaded sheets. '
                            'Check that the header includes Aadhaar, Aadhar, Adhaar or Adaar.')
    sections = []
    results = {}
    remaining = 20
    for table, column in selected.items():
        frame = tables[table]
        mask = missing_mask(frame[column])
        identifiers = [col for col in frame.columns if ''.join(normalize(col).split()) in IDENTIFIER_NAMES]
        matched = frame.loc[mask, identifiers].copy()
        # Source positions do not depend on user-supplied or duplicate index labels.
        source_name = 'Source data row'
        while source_name in matched.columns:
            source_name = '_' + source_name
        matched.insert(0, source_name, [pos for pos, missing in enumerate(mask, 1) if missing])
        results[table] = matched
        sections.append(f'**{display_value(table)}** — {len(matched):,} of {len(frame):,} records '
                        f'have missing **{display_value(column)}** values.')
        for row in matched.head(remaining).to_dict('records'):
            sections.append('- ' + '; '.join(f'{display_value(key)}: {display_value(value)}'
                                            for key, value in row.items()))
        remaining -= min(remaining, len(matched))
        if not identifiers and len(matched):
            sections.append('Student name/roll columns could not be identified; source data rows are shown.')
    if sum(len(frame) for frame in results.values()) > 20:
        sections.append('The chat lists the first 20 matches. All matching records are in the result tables and CSV downloads below.')
    sections.append('Checked every row locally. Missing includes blank/whitespace cells, nulls, and explicit '
                    'markers such as N/A or not provided. Source data rows start at 1, excluding the header; '
                    'zero is not assumed missing.')
    return DirectAnswer('\n\n'.join(sections), results)
