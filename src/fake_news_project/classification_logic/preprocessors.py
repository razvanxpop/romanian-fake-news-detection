import spacy
import re
import ast


class TextPreprocessor:
    _nlp = None
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            try:
                print("Preprocessor: Loading SpaCy ro_core_news_lg model...")
                cls._nlp = spacy.load("ro_core_news_lg")
                print("Preprocessor: SpaCy model loaded successfully.")
            except OSError:
                print("Preprocessor: ERROR - SpaCy model 'ro_core_news_lg' not found.")
                cls._nlp = None
        return cls._instance

    def __init__(self):
        if self._nlp is None and TextPreprocessor._nlp is not None:
            self._nlp = TextPreprocessor._nlp
        elif self._nlp is None and TextPreprocessor._nlp is None:
            try:
                print("Preprocessor (init): Loading SpaCy ro_core_news_lg model...")
                TextPreprocessor._nlp = spacy.load("ro_core_news_lg")
                self._nlp = TextPreprocessor._nlp
                print("Preprocessor (init): SpaCy model loaded successfully.")
            except OSError:
                print("Preprocessor (init): ERROR - SpaCy model 'ro_core_news_lg' not found during init.")
                self._nlp = None

    def get_processed_text_for_tfidf(self, text: str) -> str:
        if not self._nlp:
            print("Preprocessor: SpaCy model not available. Returning basic cleaned text.")
            text = text.lower()
            text = re.sub(r'[^a-zăâîșț\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        cleaned_text = re.sub(r'\s+', ' ', re.sub(r'[^a-zăâîșț\s]', '', text.lower())).strip()
        if not cleaned_text:
            return ""

        doc = self._nlp(cleaned_text)
        lemmas_filtered = [
            token.lemma_.lower() for token in doc
            if not token.is_stop
               and not token.is_punct
               and token.is_alpha
               and token.pos_ not in {"ADP", "CCONJ", "SCONJ"}
        ]

        return ' '.join(lemmas_filtered)
