import time

import pandas as pd
import requests


def test_topic_modeller(abstracts):
    timings = []
    for abstract in abstracts:
        start_time = time.time()
        requests.post("http://localhost:8080/classify", data={"abstract": abstract})
        end_time = time.time()
        print("Took {} second(s)".format(end_time - start_time))
        timings.append(end_time - start_time)
    average_timing = sum(timings) / len(timings)
    print("On average it took {} seconds for a request \n And it took {} seconds to process {} abstracts".format(
        average_timing, sum(timings), len(timings)))


n = 5
data = pd.read_csv("data.csv")
abstracts = data.loc[:n, "Abstract"].values

test_topic_modeller(abstracts)
