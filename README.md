
# Gen AI - Case study
The goal of this application is intended to predict the topic of open-ended questions in a market research context.
### Application structure
```
topic-modelling
 ┣ data
 ┃ ┗ technical-test-dataset.csv
 ┣ src
 ┃ ┣ model_stash: Stores model files
 ┃ ┣ __init__.py
 ┃ ┣ infer.py : Performs the inference using the trained model
 ┃ ┣ nltk_setup.py
 ┃ ┣ train.py : Trains (fine-tunes) a BERTopic model on the provided data
 ┃ ┗ utils.py : Utility functions including preprocessing and data preparation
 ┣ .dockerignore
 ┣ .gitignore
 ┣ Dockerfile
 ┣ README.md
 ┣ __init__.py
 ┣ app.py : Serves model for inference through REST API
 ┣ entrypoint.sh
 ┗ requirements.txt
```

## Build instructions

### Build with Docker (recommended) 
Given you have Docker installed on your system, build the image with:</br>```docker build -t topic_modelling .``` 
(May take several minutes)

<i>Caveat</i>: If you are on a M1/M2 Mac, add the following flag to the above command: </br>```--platform linux/arm64/v8```

Run the container with:
```docker run -e PORT=8000 -p 8000:8000 topic_modelling```
### Build with conda (alternative)
Given you have anaconda installed on your system, build, create a virtual environment and install the dependencies:
```
conda create -n topic_modelling python=3.11
conda activate topic_modelling
pip install -r requirements.txt
```
Train the model:
```
python src/train.py
```
Test the inference:
```
python src/infer.py
```
Serve the model:
```bash ./entrypoint.sh```

## Communicate with API, access the UI

Once built, you can send requests to the API by running the following:

```
curl -X POST "http://localhost:8000/predict_topic" -H "Content-Type: application/json" -d '{"text": "An open-ended answer"}'
```
(The first execution of this curl command is slower than the subsequent ones)

Access the API documentation:
```
http://localhost:8000/docs
```

Play around with a user friendly UI:
```
http://localhost:8000/
```

## Brief suggestions for improvements
 - Use two docker containers, one for training, the other for inference. When we wish to use a better model, in a CI/CD context when we, the latter helps prevents rebuilding the entire application. 
 - Run the inference container on a GPU-enabled machine.
 - Translate business needs into preprocessing / model configurations to best respond the needs. (provide a custom `--bertopic-config` to `train.py` for instance)
