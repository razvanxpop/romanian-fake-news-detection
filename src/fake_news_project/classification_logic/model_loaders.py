import os
import threading
import pickle

from gensim.models import Word2Vec

BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

MODELS_SAVED_DIR = os.path.join(BASE_PROJECT_DIR, 'models_saved')
EMBEDDINGS_DIR = os.path.join(MODELS_SAVED_DIR, 'embeddings')
CLASSIFIERS_DIR = os.path.join(MODELS_SAVED_DIR, 'classification')

DOC2VEC_MODEL_PATH = os.path.join(EMBEDDINGS_DIR, 'doc2vec_300d.model')

W2V_300D_MODEL_FILENAME = 'word2vec_300d.model'
W2V_150D_MODEL_FILENAME = 'word2vec_150d.model'

VECTORIZER_PATH = os.path.join(CLASSIFIERS_DIR, 'final_tfidf_vectorizer.pkl')
SVM_MODEL_PATH = os.path.join(CLASSIFIERS_DIR, 'final_svm_model.pkl')

class Word2VecManagerSingleton:
    _instance = None
    _lock = threading.Lock()
    _loaded_models = {}

    MODEL_CONFIG = {
        '300': {'path': os.path.join(EMBEDDINGS_DIR, W2V_300D_MODEL_FILENAME), 'name': '300D'},
        '150': {'path': os.path.join(EMBEDDINGS_DIR, W2V_150D_MODEL_FILENAME), 'name': '150D'}
    }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    print("Word2VecManagerSingleton initialized.")
        return cls._instance

    def get_model(self, dimension_key: str):
        if dimension_key not in self._loaded_models or self._loaded_models[dimension_key] is None:
            with self._lock:
                if dimension_key not in self._loaded_models or self._loaded_models[dimension_key] is None:
                    config = self.MODEL_CONFIG.get(str(dimension_key))
                    if not config:
                        print(f"Word2VecManager: No configuration found for dimension key {dimension_key}.")
                        self._loaded_models[dimension_key] = None
                        return None

                    model_path = config['path']
                    model_name_display = config['name']

                    print(f"Word2VecManager: Loading Word2Vec model ({model_name_display}) from {model_path}...")
                    if not os.path.exists(model_path):
                        print(f"Word2VecManager: ERROR - Model file not found at {model_path}")
                        self._loaded_models[dimension_key] = None
                    else:
                        try:
                            self._loaded_models[dimension_key] = Word2Vec.load(model_path)
                            print(f"Word2VecManager: Model ({model_name_display}) loaded successfully.")
                        except Exception as e:
                            print(f"Word2VecManager: Error loading Word2Vec model ({model_name_display}): {e}")
                            self._loaded_models[dimension_key] = None

        return self._loaded_models.get(dimension_key)

class TfidfSvmSingleton:
    _instance = None
    _lock = threading.Lock()

    _vectorizer = None
    _svm_model = None

    _label_mapping = {
        0: 'FAKE',
        1: 'MISINFORMATION',
        2: 'PROPAGANDA',
        3: 'REAL',
        4: 'SATIRE'
    }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("Initializing TfidfSvmSingleton...")
                    cls._instance = cls()

                    try:
                        print(f"Loading TF-IDF vectorizer from {VECTORIZER_PATH}...")
                        cls._vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
                        print("TF-IDF vectorizer loaded successfully.")
                    except FileNotFoundError:
                        print(f"ERROR: Vectorizer file not found at {VECTORIZER_PATH}")
                    except Exception as e:
                        print(f"ERROR loading vectorizer: {e}")

                    try:
                        print(f"Loading SVM model from {SVM_MODEL_PATH}...")
                        cls._svm_model = pickle.load(open(SVM_MODEL_PATH, 'rb'))
                        print("SVM model loaded successfully.")
                    except FileNotFoundError:
                        print(f"ERROR: SVM model file not found at {SVM_MODEL_PATH}")
                    except Exception as e:
                        print(f"ERROR loading SVM model: {e}")
        return cls._instance

    def get_vectorizer(self):
        return self._vectorizer

    def get_svm_model(self):
        return self._svm_model

    def get_label_mapping(self):
        return self._label_mapping