from django.apps import AppConfig

class FakeNewsUiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fake_news_ui'

    def ready(self):
        print("FakeNewsUiConfig: App ready. Preloading models and preprocessor...")
        try:
            from ..classification_logic.model_loaders import (
                TfidfSvmSingleton,
                TextPreprocessor,
                Word2VecManagerSingleton

            )

            TfidfSvmSingleton.get_instance()
            TextPreprocessor.get_instance()

            w2v_manager = Word2VecManagerSingleton.get_instance()
            w2v_manager.get_model('300')
            w2v_manager.get_model('150')

            print("FakeNewsUiConfig: Models preloaded/initialized via Singletons.")
        except Exception as e:
            print(f"FakeNewsUiConfig: Error during model preloading: {e}")