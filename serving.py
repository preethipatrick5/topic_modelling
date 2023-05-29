import numpy as np
import pandas as pd

import bert
import utils

model = None


def get_model():
    global model
    model = bert.BertToSingleLayerNeuralNetwork(config=bert.ModelConfig)
    model.version = "final"
    model.build(processors=[utils.replace_latex_math_with, utils.to_corpus, utils.lemmatize_sentence])
    model.load("model")


def get_res(abstract):
    global labels
    global model
    labels = np.array(
        ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology", "Quantitative Finance"])
    data = pd.DataFrame({"text": [abstract]})
    y_pred = model.predict(data["text"])[0]
    print("y_pred:", y_pred)
    top_labels = (y_pred > 0.5).astype(int)
    selected_labels = labels[np.where(top_labels == 1)]
    response = {
        "status": 200, "body": {
            "abstract": abstract,
            "labels": selected_labels.tolist(),
            "classification": top_labels,
            "confidenceScores": {
                "Computer": float(y_pred[np.where(labels == "Computer Science")][0]),
                "Physics": float(y_pred[np.where(labels == "Physics")][0]),
                "Maths": float(y_pred[np.where(labels == "Mathematics")][0]),
                "Statistics": float(y_pred[np.where(labels == "Statistics")][0]),
                "Biology": float(y_pred[np.where(labels == "Quantitative Biology")][0]),
                "Finance": float(y_pred[np.where(labels == "Quantitative Finance")][0])
            }}}
    return response


def main(abstract):
    # get_model()
    result = get_res(abstract)
    body = result['body']
    return body


get_model()
