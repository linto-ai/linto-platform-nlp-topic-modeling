import pandas as pd
from spacy.tokens import Doc
from bertopic import BERTopic

class TopicModeler:
    """
    Wrapper class for BERTopic.
    """
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        if not Doc.has_extension("topics"):
            Doc.set_extension("topics", default=[])
        if not Doc.has_extension("topic_assignments"):
            Doc.set_extension("topic_assignments", default=[])

    def __call__(self, doc, **kwargs):
        runtime_kwargs = {}
        runtime_kwargs.update(self.kwargs)
        runtime_kwargs.update(kwargs)

        topic_model = BERTopic(language=None, calculate_probabilities=True, embedding_model=self.model, **runtime_kwargs)

        segments = [seg.text.split('|')[0].strip() for seg in doc.sents]
        # (semi)-Supervised Topic Modeling: providing ground truth labels to ".fit_transform(y=labels)" is not integrated.
        # https://maartengr.github.io/BERTopic/tutorial/supervised/supervised.html
        predictions, probabilities = topic_model.fit_transform(segments)
        
        topics = topic_model.get_topic_freq()
        topics["Name"] = topics.Topic.map(topic_model.get_topics())
        doc._.topics = topics.to_dict("records")

        topic_assignments = pd.DataFrame(data=predictions, columns=["Prediction"])
        topic_assignments["Probabilities"] = probabilities.tolist()
        topic_assignments["Segment"] = segments
        doc._.topic_assignments = topic_assignments.to_dict("records")

        return doc
