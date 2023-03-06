import sys

input_file = sys.argv[1]
num_beams = int(sys.argv[2])
output_file = sys.argv[3]

document = [line.strip() for line in open(input_file)]


documents_extended = []
for doc in document:
    for _ in range(num_beams):
        documents_extended.append(doc)

with open(output_file,"w") as f:
    for doc in documents_extended:
        f.write(doc + "\n")
