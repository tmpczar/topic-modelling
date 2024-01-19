import os
from bertopic import BERTopic
from typing import Dict
from utils import preprocess_data, format_topic_string

model_location = "./model_stash"
topic_model = BERTopic.load(model_location)

topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
topic_info["Name"] = topic_info["Name"].apply(format_topic_string)
# topic info maps topic id to topic string
topic_info: Dict[int, str] = dict(topic_info.values)


def predict(x: str) -> dict:
    """
        Predicts a topic from the fine-tuned BERTopic model
    Args:
        x (str): An open-ended answer to find a topic for.

    Returns:
        dict: A human-readable topic with the probability of its prediction.
    """

    x = preprocess_data(x)

    topics, probs = topic_model.transform(x)

    # Only output most probable topic
    predicted_topic = topics[0]
    confidence = probs[0]

    result = {
        "predicted_topic_id": predicted_topic,
        "predicted_topic": topic_info[predicted_topic],
        "confidence": confidence,
    }
    if predicted_topic == -1:
        # -1 refers to all outliers and should typically be ignored (BERTopic docs)
        result["predicted_topic"] = "no topic found"
    return result


if __name__ == "__main__":
    print(model_location)

    res = predict("this is gibberish")
    print(res)
