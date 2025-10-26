from django import forms

class NewsArticleForm(forms.Form):
    news_content = forms.CharField(
        label="News Article",
        widget=forms.Textarea(attrs={
            'placeholder': "Write or paste the news content here...",
            'rows': 10,
            'cols': 80,
            'class': 'form-control'
        }),
        required=True,
        help_text="Enter the full text of the article."
    )

class WordSimilarityForm(forms.Form):
    target_word = forms.CharField(
        label="Target Word (Lemma)",
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter the word'})
    )
    MODEL_DIMENSION_CHOICES = [
        ('300', 'Word2Vec 300D'),
        ('150', 'Word2Vec 150D'),
    ]
    model_dimension = forms.ChoiceField(
        label="Select Word2Vec Model",
        choices=MODEL_DIMENSION_CHOICES,
        initial='300',
        widget=forms.Select(attrs={'class': 'form-control-sm'})
    )
    top_n = forms.IntegerField(
        label="Number of Neighbors (Top N)",
        min_value=1,
        max_value=50,
        initial=10,
        required=True,
        widget=forms.NumberInput(attrs={'class': 'form-control-sm'})
    )