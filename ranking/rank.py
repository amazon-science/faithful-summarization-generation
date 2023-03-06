import sys
import json
import pandas as pd

file_prefix = sys.argv[1]

num_beams = int(sys.argv[2])

document = [line.strip() for line in open(file_prefix + ".document")]
summary = [line.strip() for line in open(file_prefix + ".summary")]

id = []
for i in range(len(document)):
    id.append(i//num_beams)

factcc = json.load(open(file_prefix + "_factcc.json"))
dae = json.load(open(file_prefix + "_dae.json"))
bsfact = json.load(open(file_prefix + "_dae.json"))
questeval = json.load(open(file_prefix + "_questeval.json"))

# create csv

d = {
    "id": id,
    "document": document,
    "summary": summary,
    "questeval":questeval,
    "dae":dae,
    "factcc":factcc,
    "bsfact":bsfact,
}

print({k:len(v) for k,v in d.items()})

df = pd.DataFrame.from_dict(d)

print(df)

df.to_csv(file_prefix + ".csv", index=False)

# rank
models = ["bsfact", "factcc","dae", "questeval"]
weight, bias = [1.96576989, 0.2972612, -0.29037403,  0.93960678], -1.9096430327732379

df["composite"] = [sum( [row[m] * w for m,w in zip(models, weight)] ) + bias for i, row in df.iterrows()]

summ = df.loc[df.groupby(["id"])["composite"].idxmax()]["summary"]

with open(file_prefix + "_ranked_summary.txt", "w") as f:
    for s in summ:
        f.write(s + "\n")