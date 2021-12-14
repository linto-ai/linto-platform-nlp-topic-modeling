from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Article(BaseModel):
    # Schema for a single article in a batch of articles to process
    text: str


class RequestModel(BaseModel):
    # Schema for a request consisting a batch of articles, and component configuration
    articles: List[Article]
    component_cfg: Optional[Dict[str, Dict[str, Any]]] = None


class TopicResponseModel(BaseModel):
    # This is the schema of the expected response and depends on what you
    # return from get_data.

    class Batch(BaseModel):
        class Topic(BaseModel):
            class Phrase(BaseModel):
                text: str
                score: float
            topic_id: int
            count: int
            phrases: List[Phrase] = []

        class Assignment(BaseModel):
            text: str
            assigned_id: int
            probabilities: List[float] = []

        text: str
        topics: List[Topic] = []
        topic_assignments: List[Assignment] = []

    topic: List[Batch]
