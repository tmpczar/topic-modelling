import click  # idea is to make it modular, i.e put any HuggingFace model you want
from bertopic import BERTopic
from typing import List
import pandas as pd
import json
import logging
from utils import prepare_data, preprocess_data
import os 

def parse_bertopic_config(ctx, param, value):
    if value is None:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise click.BadParameter("Invalid JSON format for --bertopic-config")

@click.command()
@click.option(
    "--data-filepath",
    default=f"data/technical-test-dataset.csv",
    help="Location of the data (.csv file)",
)
@click.option(
    "--model-save-filepath",
    default=f"./model_stash",
    help="Path to save the BERTopic model",
)
@click.option(
    "--bertopic-config",
    default={},
    help="Dictionary of BERTopic configuration options as JSON",
    type=click.STRING,  # Force input to be treated as a string
    callback=parse_bertopic_config,  # Use the custom type conversion function
)
def main(data_filepath, model_save_filepath, bertopic_config):
    """Trains the BERTopic model on the provided dataset and saves it.
    The configuration of the BERTopic model is fully customizable, by providing a JSON file of the desired parameters.
    """
    try:
        docs = prepare_data(filepath=data_filepath)
        docs = [preprocess_data(x) for x in docs]
        logging.info(f"Data processed")
    except Exception as E:
        logging.error(f"Could not process the data provided: {E}")

    try:
        bertopic_config["verbose"] = True
        topic_model = BERTopic(**bertopic_config)

        topic_model.fit_transform(docs)
        logging.info(f"Model has trained")
    except Exception as E:
        logging.error(f"Could not train the model: {E}")

    try:
        # Saving to safetensors for size purposes:
        # https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.reduce_topics
        topic_model.save(
            path=model_save_filepath, serialization="safetensors", save_ctfidf=True
        )
        logging.info(f"Model saved at: {model_save_filepath}")
    except Exception as E:
        logging.error(f"Could not save the model: {E}")


if __name__ == "__main__":
    main()
