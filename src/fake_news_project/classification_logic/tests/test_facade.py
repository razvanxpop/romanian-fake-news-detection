import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from ..facade import NewsClassifierFacade


class TestNewsClassifierFacade(unittest.TestCase):
    @patch('classification_logic.facade.TextPreprocessor.get_instance')
    @patch('classification_logic.facade.TfidfSvmSingleton.get_instance')
    def test_classify_high_confidence(self, mock_get_singleton, mock_get_preprocessor):
        mock_svm_model = MagicMock()
        mock_vectorizer = MagicMock()

        mock_svm_model.predict_proba.return_value = np.array([[0.05, 0.92, 0.01, 0.01, 0.01]])

        mock_singleton_instance = MagicMock()
        mock_singleton_instance.get_svm_model.return_value = mock_svm_model
        mock_singleton_instance.get_vectorizer.return_value = mock_vectorizer
        mock_singleton_instance.get_label_mapping.return_value = {1: 'fake_news'}
        mock_get_singleton.return_value = mock_singleton_instance

        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.get_processed_text_for_tfidf.return_value = "processed text"
        mock_get_preprocessor.return_value = mock_preprocessor_instance

        facade = NewsClassifierFacade()

        result = facade.classify("This is some fake news.")

        self.assertEqual(result['classification_result'], 'fake_news')
        self.assertAlmostEqual(result['confidence'], 0.92)
        self.assertNotIn('all_predictions', result)

    @patch('classification_logic.facade.TextPreprocessor.get_instance')
    @patch('classification_logic.facade.TfidfSvmSingleton.get_instance')
    def test_classify_medium_confidence(self, mock_get_singleton, mock_get_preprocessor):
        mock_svm_model = MagicMock()
        mock_vectorizer = MagicMock()

        mock_svm_model.predict_proba.return_value = np.array([[0.15, 0.75, 0.05, 0.03, 0.02]])

        mock_singleton_instance = MagicMock()
        mock_singleton_instance.get_svm_model.return_value = mock_svm_model
        mock_singleton_instance.get_vectorizer.return_value = mock_vectorizer

        mock_singleton_instance.get_label_mapping.return_value = {
            0: 'real_news',
            1: 'misinformation',
            2: 'propaganda',
            3: 'fake_news',
            4: 'satire'
        }
        mock_get_singleton.return_value = mock_singleton_instance

        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.get_processed_text_for_tfidf.return_value = "processed text"
        mock_get_preprocessor.return_value = mock_preprocessor_instance

        facade = NewsClassifierFacade()
        result = facade.classify("This is some misinformation.")

        self.assertEqual(result['classification_result'], 'misinformation')
        self.assertAlmostEqual(result['confidence'], 0.75)
        self.assertIn('all_predictions', result)

        self.assertEqual(len(result['all_predictions']), 5)

        self.assertEqual(result['all_predictions'][0][0], 'misinformation')
        self.assertEqual(result['all_predictions'][0][1], 0.75)

    @patch('classification_logic.facade.TextPreprocessor.get_instance')
    @patch('classification_logic.facade.TfidfSvmSingleton.get_instance')
    def test_classify_low_confidence(self, mock_get_singleton, mock_get_preprocessor):
        mock_svm_model = MagicMock()
        mock_vectorizer = MagicMock()
        mock_svm_model.predict_proba.return_value = np.array([[0.30, 0.28, 0.15, 0.14, 0.13]])

        mock_singleton_instance = MagicMock()
        mock_singleton_instance.get_svm_model.return_value = mock_svm_model
        mock_singleton_instance.get_vectorizer.return_value = mock_vectorizer
        mock_singleton_instance.get_label_mapping.return_value = {0: 'real_news'}
        mock_get_singleton.return_value = mock_singleton_instance

        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.get_processed_text_for_tfidf.return_value = "processed text"
        mock_get_preprocessor.return_value = mock_preprocessor_instance

        facade = NewsClassifierFacade()
        result = facade.classify("This is some ambiguous news.")

        self.assertEqual(result['classification_result'], 'MANUAL_VERIFICATION')
        self.assertAlmostEqual(result['confidence'], 0.30)
        self.assertNotIn('all_predictions', result)

    @patch('classification_logic.facade.TextPreprocessor.get_instance')
    @patch('classification_logic.facade.TfidfSvmSingleton.get_instance')
    def test_model_not_loaded_error(self, mock_get_singleton, mock_get_preprocessor):
        mock_singleton_instance = MagicMock()
        mock_singleton_instance.get_svm_model.return_value = None
        mock_singleton_instance.get_vectorizer.return_value = None
        mock_get_singleton.return_value = mock_singleton_instance

        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.get_processed_text_for_tfidf.return_value = "processed text"
        mock_get_preprocessor.return_value = mock_preprocessor_instance

        facade = NewsClassifierFacade()
        result = facade.classify("Some text.")

        self.assertEqual(result['classification_result'], 'ERROR')
        self.assertIn('Model components are not available', result.get('message', ''))


if __name__ == '__main__':
    unittest.main()
