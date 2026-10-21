import io
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from services.direct_queries import answer_missing_aadhaar
from test_missing_values import TABLE, student_tables


class DirectQueryTests(unittest.TestCase):
    def test_reported_questions_and_spellings(self):
        for question in ['Which Students does not have an Adaar ?',
                         'Is there any student not give there Adhaar Number ?',
                         "Which students don't have Aadhar?",
                         'List all students without Aadhaar.',
                         'How many students are missing Aadhaar?']:
            with self.subTest(question=question):
                result = answer_missing_aadhaar(question, student_tables())
                self.assertIsNotNone(result)
                self.assertIn('12 of 43', result.answer)
                self.assertIn('Student 12', result.answer)
                self.assertEqual(len(result.results[TABLE]), 12)
                self.assertNotIn('Adhaar Number', result.results[TABLE].columns)

    def test_no_missing_is_verified_not_invented(self):
        result = answer_missing_aadhaar('Which students lack Adaar?', student_tables())
        self.assertIsNone(result)  # Unrecognized wording stays with the general agent.
        tables = {'all-provided': pd.DataFrame({'Student Name': ['One'], 'Aadhaar': ['123456789012']})}
        result = answer_missing_aadhaar('Which students do not have Aadhaar?', tables)
        self.assertIn('0 of 1', result.answer)

    def test_does_not_ignore_additional_conditions(self):
        for question in ['Students without Aadhaar in Section A',
                         'Students with missing Aadhaar and unpaid fees',
                         'Students without Aadhaar or email',
                         'Students with no missing Aadhaar',
                         'Which students have valid Aadhaar?',
                         'Why do students not have Aadhaar?']:
            self.assertIsNone(answer_missing_aadhaar(question, student_tables()), question)

    def test_ambiguous_columns_ask_for_exact_column(self):
        tables = {'test': pd.DataFrame({'Aadhar Number': [None], 'Aadhaar Card': [None]})}
        result = answer_missing_aadhaar('Which students do not have Adaar?', tables)
        self.assertIn('Multiple Aadhaar columns', result.answer)
        self.assertFalse(result.results)
        result = answer_missing_aadhaar('Which students do not have Aadhaar Card?', tables)
        self.assertIn('1 of 1', result.answer)

    def test_missing_header_has_clear_guidance(self):
        result = answer_missing_aadhaar('Which students do not have Aadhaar?',
                                       {'test': pd.DataFrame({'Name': ['One']})})
        self.assertIn('could not identify an Aadhaar column', result.answer)

    def test_all_10000_matches_available_but_chat_is_bounded(self):
        result = answer_missing_aadhaar('Which students do not have Aadhaar?',
            {'large': pd.DataFrame({'Name': [f'Person {i}' for i in range(10000)], 'Aadhaar': [None] * 10000})})
        self.assertEqual(len(result.results['large']), 10000)
        self.assertIn('10,000 of 10,000', result.answer)
        self.assertIn('first 20', result.answer)
        self.assertLess(len(result.answer), 6000)
        self.assertEqual(result.results['large'].iloc[-1]['Source data row'], 10000)

    def test_exact_table_scope_does_not_include_other_sheets(self):
        tables = student_tables()
        tables['CA-12.10.xlsx :: Sheet2'] = tables[TABLE].copy()
        result = answer_missing_aadhaar(f'Which students do not have Aadhaar in {TABLE}?', tables)
        self.assertEqual(list(result.results), [TABLE])

    def test_ui_answers_reported_question_without_model_or_api_key(self):
        upload = io.BytesIO()
        student_tables()[TABLE].to_excel(upload, index=False)
        upload.name, upload.size, upload.file_id = 'CA-12.10.xlsx', len(upload.getvalue()), 'direct-test'
        with (patch('streamlit.file_uploader', return_value=[upload]),
             patch('agent.factory.build_model', side_effect=AssertionError('Must not call the model')) as model,
              patch('services.analysis.analyze', side_effect=AssertionError('Must not invoke tools')) as agent):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
            app.chat_input[0].set_value('Which Students does not have an Adaar ?').run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertIn('12 of 43', app.session_state['messages'][-1]['content'])
            self.assertEqual(len(app.session_state['direct_results'][TABLE]), 12)
            model.assert_not_called()
            agent.assert_not_called()
