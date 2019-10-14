import unittest
import functions as fs
import numpy as np

class TestPreprocessing(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(fs.preprocessing('HorSe.'), 'horse .')
        self.assertEqual(fs.preprocessing('This is a Test. UPPER CASE should not Exist.'),
                         'this is a test . upper case should not exist .')

    def test_acronym(self):
        self.assertEqual(fs.preprocessing('u.s.a.'),'usa')

    def test_paranthesis(self):
        self.assertEqual(fs.preprocessing('This is an article with (text in parenthesis)'),
                         'this is an article with text in parenthesis')
    def test_brackets(self):
        self.assertEqual(fs.preprocessing('This is an article with [text in brackets]'),
                         'this is an article with text in brackets')
    def test_double_quotes(self):
        self.assertEqual(fs.preprocessing('This is an article with "double" "quotes"'),
                         'this is an article with double quotes')

    def test_single_quotes(self):
        self.assertEqual(fs.preprocessing("This is an article with 'single' 'quotes'"),
                         'this is an article with single quotes')

    def test_possessive(self):
        self.assertEqual(fs.preprocessing("This is Joe's book"),
                         'this is joe book')

    def test_space_before_punctuation(self):
        self.assertEqual(fs.preprocessing(''' Sometimes
        Phrases are not finished with period
        And got loose'''),
                         'sometimes . phrases are not finished with period . and got loose')

    def test_multiple_spaces(self):
        self.assertEqual(fs.preprocessing('Multiple     spaces should disappear!'),
                         'multiple spaces should disappear !')

class TestTreatmentOfPeriod(unittest.TestCase):

    def test_word(self):
        self.assertEqual(fs.treatment_of_period('word'),'word')

    def test_word_with_period(self):
        self.assertEqual(fs.treatment_of_period('word.'), 'word .')

    def test_acronym(self):
        self.assertEqual(fs.treatment_of_period('u.s.a.'), 'usa')

    def test_url(self):
        self.assertEqual(fs.treatment_of_period('www.lopasso.tech'), 'www.lopasso.tech')

class TestUniquify(unittest.TestCase):

    def test_repetition(self):
        self.assertEqual(fs.uniquify('This should eliminate repetition repetition for all'),
                         'This should eliminate repetition for all')

    def test_complex_repetition(self):
        self.assertEqual(fs.uniquify('Even Joe Smith, Joe Smith, should go'),
                     'Even Joe Smith, should go')

class TestRemoveSimilarSentences(unittest.TestCase):

    def test_similar_sentences(self):
        input = ['According to police, they were arrested',
                 'According to police, they were arrested yesterday']
        self.assertEqual(fs.remove_similar_sentences(input),['According to police, they were arrested yesterday'])


class TestPostProcessing(unittest.TestCase):

    def test_remove_tokens(self):
        global START_DECODING, STOP_DECODING, OOV_DECODING
        input = 'Sentences might have [PAD], [STOP] and [UNK] tokens.'
        self.assertEqual(fs.post_processing(input),'Sentences might have, and tokens.')

    def test_fix_sentences(self):
        global START_DECODING, STOP_DECODING, OOV_DECODING
        input = 'fix sentences to have punctuation .'
        self.assertEqual(fs.post_processing(input),'Fix sentences to have punctuation.')

class TestSplitter(unittest.TestCase):

    def test_long_article(self):
        sentence = 'This is a sentence with 10 words that repeats itself. '
        article = sentence*50
        self.assertEqual(len(fs.splitter(article)),2)

if __name__ == '__main__':
    unittest.main()
