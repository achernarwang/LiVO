import os
import json
from pathlib import Path
from tqdm import tqdm
from retriever import ValueRetriever

retriever = ValueRetriever()

data_path = Path('../livo_eval_data')

file_list = ["career.jsonl", "goodness.jsonl", "badness.jsonl", "nudity.jsonl", "bloody.jsonl", "zombie.jsonl"]

for file in file_list:
    with open(data_path / file, 'r') as f:
        data = [json.loads(line) for line in f]

    basename = os.path.splitext(os.path.basename(file))[0]

    with open(data_path / f'retrieved_{basename}.jsonl', 'w') as f:
        for d in tqdm(data, ncols=80):
            d["value"] = retriever.retrieve(d["prompt"])
            f.write(json.dumps(d) + '\n')