from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scorer import BERTScoreScorer
from lookahead import Lookahead
from generation import Generator
from tqdm import tqdm

import json

import argparse

parser = argparse.ArgumentParser()

# base decoding model
parser.add_argument("--model_name", type=str, default="facebook/bart-large-xsum")
parser.add_argument("--cache_dir", type=str, default="./cache")

# input output
parser.add_argument("--document_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)

# base decoding configuration. Please refer to Huggingface's GenerationMixin for the explaination of the parameters
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--max_input_length", type=int, default=512)
parser.add_argument("--max_output_length", type=int, default=64)
parser.add_argument("--do_sample", action='store_true', default=False)

# lookahead configuration
parser.add_argument("--do_lookahead", action="store_true", default=False)
parser.add_argument("--lookahead_length", type=int, default=64)
parser.add_argument("--lookahead_lambda", type=int, default=25)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--lookahead_decoding_type", type=str, default="greedy", choices=["greedy","beam","sample"])
parser.add_argument("--lookahead_beam", type=int, default=1)

# scorer configuration
parser.add_argument("--scorer_model_type", type=str, default="roberta-large")
parser.add_argument("--scorer_num_layers", type=int, default=17)

args = parser.parse_args()

# loading model
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = model.cuda() # can optionally call .half() for mixed precision

# loading input
documents = [line.strip() for line in open(args.document_file)]

# Load scorer for lookahead
scorer = BERTScoreScorer(
    model_name=args.scorer_model_type,
    num_layers=args.scorer_num_layers,
    cache_dir=args.cache_dir,
)

#  Create lookahead
lookahead = None
if args.do_lookahead:
    lookahead = Lookahead(
        model,
        tokenizer,
        scorer,
        lookahead_length=args.lookahead_length,
        lookahead_lambda=args.lookahead_lambda,
        lookahead_top_k=args.top_k,
        decoding_type=args.lookahead_decoding_type,
        num_beams=args.lookahead_beam,
        num_return_sequences=args.lookahead_beam,
        max_length=args.max_output_length,
    )

# Create generator with lookahead
generator = Generator(model, lookahead=lookahead)

summaries = []

for i in tqdm(range(0, len(documents), args.batch_size)):
    input_str = documents[i:i+args.batch_size]
    
    #  IMPROTANT! Need to prepare document
    if generator.lookahead is not None:
        generator.lookahead.scorer.prepare_document(input_str)

    inputs = tokenizer(input_str, max_length=args.max_input_length, padding=True, truncation=True, return_tensors="pt")

    inputs = {k:v.cuda() for k,v in inputs.items()}

    output = generator.generate(
        input_ids = inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_length=args.max_output_length,
        do_sample=args.do_sample,
    )

    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    if args.num_return_sequences == 1:
        summaries += output
    else:
        for i in range(0, len(output), args.num_return_sequences):
            summaries.append(output[i:i+args.num_return_sequences])

# Save file
with open(args.output_file, "w") as f:
    if args.num_return_sequences == 1:
        for line in summaries:
            f.write(line + "\n")
    else:
        json.dump(summaries, f)