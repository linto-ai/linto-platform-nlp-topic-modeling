from typing import Dict, Any

from spacy.tokens import Doc

def get_data(doc: Doc) -> Dict[str, Any]:
    """Extract the data to return from the REST API given a Doc object. Modify
    this function to include other data."""
    topics = [
        {
            "topic_id": entry["Topic"],
            "count": entry["Count"],
            "phrases": [
                {
                    "text": phrase[0],
                    "score": phrase[1]
                } 
                for phrase in entry["Name"]
            ]
        }
        for entry in doc._.topics
    ]

    topic_assignments = [
        {
            "text": entry["Segment"],
            "assigned_id": entry["Prediction"],
            "probabilities": entry["Probabilities"]
        }
        for entry in doc._.topic_assignments
    ]
    return {"text": doc.text, "topics": topics, "topic_assignments": topic_assignments}
    