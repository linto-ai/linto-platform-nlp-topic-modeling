import spacy
import hdbscan
from spacy.language import Language
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
from thinc.api import Config
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from components.topic_modeler import TopicModeler

# Load components' defaut configuration
config = Config().from_disk("components/config.cfg")

@Language.factory("topic", default_config=config["components"]["topic"])
def make_topic_modeler(
    nlp: Language,
    name: str,
    model: SentenceTransformer,
    top_n_words: int = 10,
    n_gram_range: Tuple[int, int] = (1, 1),
    min_topic_size: int = 3,
    nr_topics: Union[int, str] = None,
    low_memory: bool = False,
    seed_topic_list: List[List[str]] = None,
    umap_model: UMAP = None,
    hdbscan_model: hdbscan.HDBSCAN = None,
    vectorizer_model: CountVectorizer = None,
    verbose: bool = False
    ):

    kwargs = locals()
    del kwargs['nlp']
    del kwargs['name']
    del kwargs['model']

    return TopicModeler(model, **kwargs)

