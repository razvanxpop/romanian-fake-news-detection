from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch, MagicMock

from .forms import NewsArticleForm, WordSimilarityForm


class FormTests(TestCase):
    def test_news_article_form_valid(self):
        form = NewsArticleForm(data={'news_content': 'This is valid content.'})
        self.assertTrue(form.is_valid())

    def test_news_article_form_invalid_empty(self):
        form = NewsArticleForm(data={'news_content': ''})
        self.assertFalse(form.is_valid())
        self.assertIn('news_content', form.errors)

    def test_word_similarity_form_valid(self):
        form = WordSimilarityForm(data={'target_word': 'test', 'model_dimension': '300', 'top_n': 10})
        self.assertTrue(form.is_valid())

    def test_word_similarity_form_invalid_data(self):
        form = WordSimilarityForm(data={'model_dimension': '300', 'top_n': 10})
        self.assertFalse(form.is_valid())
        self.assertIn('target_word', form.errors)

        form = WordSimilarityForm(data={'target_word': 'test', 'model_dimension': '300', 'top_n': 100})
        self.assertFalse(form.is_valid())
        self.assertIn('top_n', form.errors)

        form = WordSimilarityForm(data={'target_word': 'test', 'model_dimension': '999', 'top_n': 10})
        self.assertFalse(form.is_valid())
        self.assertIn('model_dimension', form.errors)

class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.index_url = reverse('fake_news_ui:index')
        self.classify_url = reverse('fake_news_ui:classify_article')
        self.similarity_url = reverse('fake_news_ui:word_similarity')

    @patch('fake_news_ui.views.classifier_facade.classify')
    def test_classify_view_post_high_confidence(self, mock_classify):
        mock_classify.return_value = {'classification_result': 'FAKE', 'confidence': 0.92}

        response = self.client.post(self.classify_url, {'news_content': 'Some fake news'})

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fake_news_ui/result.html')
        mock_classify.assert_called_once_with('Some fake news')
        self.assertEqual(response.context['classification_result'], 'FAKE')

        self.assertNotIn('MANUAL_VERIFICATION', response.content.decode())

    @patch('fake_news_ui.views.classifier_facade.classify')
    def test_classify_view_post_medium_confidence(self, mock_classify):
        mock_classify.return_value = {
            'classification_result': 'MISINFORMATION',
            'confidence': 0.75,
            'all_predictions': [('MISINFORMATION', 0.75), ('FAKE', 0.15), ('REAL', 0.10)]
        }

        response = self.client.post(self.classify_url, {'news_content': 'Some medium news'})

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fake_news_ui/result.html')
        mock_classify.assert_called_once_with('Some medium news')
        self.assertIn('all_predictions', response.context)

        self.assertIn("Top Prediction:", response.content.decode())
        self.assertIn("All Predictions:", response.content.decode())

    @patch('fake_news_ui.views.classifier_facade.classify')
    def test_classify_view_post_low_confidence(self, mock_classify):
        mock_classify.return_value = {'classification_result': 'MANUAL_VERIFICATION', 'confidence': 0.45}

        response = self.client.post(self.classify_url, {'news_content': 'Some ambiguous news'})

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fake_news_ui/result.html')
        mock_classify.assert_called_once_with('Some ambiguous news')
        self.assertEqual(response.context['classification_result'], 'MANUAL_VERIFICATION')

        self.assertIn("Manual Verification Required", response.content.decode())

    def test_classify_view_post_invalid_form(self):
        response = self.client.post(self.classify_url, {'news_content': ''})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fake_news_ui/index.html')
        self.assertTrue(response.context['form'].errors)

    def test_index_view_get(self):
        response = self.client.get(self.index_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fake_news_ui/index.html')
        self.assertIsInstance(response.context['form'], NewsArticleForm)

    def test_word_similarity_view_get(self):
        response = self.client.get(self.similarity_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fake_news_ui/word_similarity.html')
        self.assertIsInstance(response.context['form'], WordSimilarityForm)

    @patch('fake_news_ui.views.word2vec_manager.get_model')
    def test_word_similarity_view_post_word_found(self, mock_get_model):
        mock_w2v_model = MagicMock()
        mock_w2v_model.wv.most_similar.return_value = [('neighbor1', 0.9), ('neighbor2', 0.8)]
        mock_w2v_model.wv.key_to_index = {'test': 0}
        mock_get_model.return_value = mock_w2v_model

        form_data = {'target_word': 'test', 'model_dimension': '300', 'top_n': 2}
        response = self.client.post(self.similarity_url, form_data)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fake_news_ui/word_similarity.html')
        self.assertIn('similar_words', response.context)
        self.assertEqual(len(response.context['similar_words']), 2)

    @patch('fake_news_ui.views.word2vec_manager.get_model')
    def test_word_similarity_view_post_word_not_found(self, mock_get_model):
        mock_w2v_model = MagicMock()
        mock_w2v_model.wv.most_similar.side_effect = KeyError("Word not in vocab")
        mock_w2v_model.wv.key_to_index = {'other_word': 0}
        mock_get_model.return_value = mock_w2v_model

        form_data = {'target_word': 'mere', 'model_dimension': '300', 'top_n': 10}
        response = self.client.post(self.similarity_url, form_data)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fake_news_ui/word_similarity.html')
        self.assertIn('error_message', response.context)
        self.assertNotIn('similar_words', response.context)
        self.assertIn("The word 'mere' was not found in the vocabulary.",
                      response.context['error_message'])

