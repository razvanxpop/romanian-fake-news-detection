from django.shortcuts import render
from django.views.generic import FormView
from django.urls import reverse_lazy

from classification_logic.facade import NewsClassifierFacade
from .forms import NewsArticleForm, WordSimilarityForm
from classification_logic.model_loaders import Word2VecManagerSingleton

classifier_facade = NewsClassifierFacade()
word2vec_manager = Word2VecManagerSingleton.get_instance()


class IndexView(FormView):
    template_name = 'fake_news_ui/index.html'
    form_class = NewsArticleForm

    def form_valid(self, form):
        news_content = form.cleaned_data['news_content']

        result_data = classifier_facade.classify(news_content)

        context = {'submitted_content': news_content}
        context.update(result_data)

        return render(self.request, 'fake_news_ui/result.html', context)


class WordSimilarityView(FormView):
    template_name = 'fake_news_ui/word_similarity.html'
    form_class = WordSimilarityForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

    def form_valid(self, form):
        target_word = form.cleaned_data['target_word'].lower().strip()
        model_dim_key = form.cleaned_data['model_dimension']
        top_n = form.cleaned_data['top_n']

        context = self.get_context_data()
        context['submitted_word'] = target_word
        context['selected_dimension'] = f"{model_dim_key}D"

        w2v_model = word2vec_manager.get_model(model_dim_key)

        if w2v_model:
            try:
                similar_words = w2v_model.wv.most_similar(target_word, topn=top_n)
                context['similar_words'] = similar_words
                context['vocabulary_size'] = len(w2v_model.wv.key_to_index)
            except KeyError:
                context['error_message'] = f"The word '{target_word}' was not found in the vocabulary."
            except Exception as e:
                context['error_message'] = f"An error occurred: {str(e)}"
        else:
            context['error_message'] = f"The Word2Vec {model_dim_key}D model could not be loaded."

        return self.render_to_response(context)