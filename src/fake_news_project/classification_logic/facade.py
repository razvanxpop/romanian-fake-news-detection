import os

from .model_loaders import TfidfSvmSingleton

from .preprocessors import TextPreprocessor


class NewsClassifierFacade:
    def __init__(self):
        print("NewsClassifierFacade: Initializing...")
        model_singleton = TfidfSvmSingleton.get_instance()
        self.vectorizer = model_singleton.get_vectorizer()
        self.svm_model = model_singleton.get_svm_model()
        self.label_mapping = model_singleton.get_label_mapping()
        self.preprocessor = TextPreprocessor.get_instance()

        if not self.vectorizer or not self.svm_model:
            print("NewsClassifierFacade: WARNING - Vectorizer or SVM model not loaded.")
        print("NewsClassifierFacade: Initialization complete.")

    def classify(self, text: str) -> dict:
        if not text or not text.strip():
            return {'classification_result': "ERROR", 'message': "No content submitted."}

        if not self.vectorizer or not self.svm_model:
            return {'classification_result': "ERROR", 'message': "Model components are not available."}

        try:
            processed_text = self.preprocessor.get_processed_text_for_tfidf(text)
            if not processed_text:
                return {'classification_result': 'MANUAL_VERIFICATION', 'confidence': 0.0,
                        'message': 'Text has no content after preprocessing. Needs manual check.'}

            text_vector = self.vectorizer.transform([processed_text])

            probabilities = self.svm_model.predict_proba(text_vector)[0]

            confidence = max(probabilities)
            top_class_index = probabilities.argmax()
            top_label = self.label_mapping.get(top_class_index, "UNKNOWN")

            result_data = {'confidence': confidence}

            if confidence > 0.80:
                result_data['classification_result'] = top_label
            elif 0.60 <= confidence <= 0.80:
                result_data['classification_result'] = top_label
                all_probs = {self.label_mapping.get(i, "UNKNOWN"): prob
                             for i, prob in enumerate(probabilities)}
                result_data['all_predictions'] = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
            else:
                result_data['classification_result'] = 'MANUAL_VERIFICATION'

            return result_data

        except Exception as e:
            print(f"Facade: Error during classification - {str(e)}")
            return {'classification_result': "ERROR", 'message': f"An error occurred during processing."}
