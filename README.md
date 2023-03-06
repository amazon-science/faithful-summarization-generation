# Faithfulness-Aware Decoding Strategies for Abstractive Summarization (EACL 2023)

This repository contains the code for the paper "Faithfulness-Aware Decoding Strategies for Abstractive Summarization."

**Authors:** [David Wan](https://meetdavidwan.github.io), [Mengwen Liu](https://www.amazon.science/author/mengwen-liu), [Kathleen McKeown](http://www.cs.columbia.edu/~kathy), [Markus Dreyer](https://markusdreyer.org), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal)

**Arxiv:** SOON!

## 1. Generating Summaries with Lookahead

### 1.1. Environment
Needed packages:
- PyTorch
- [transformers](https://github.com/huggingface/transformers/tree/v4.21.0) >= 4.21.0
- [datasets](https://github.com/huggingface/datasets/tree/2.4.0) >=2.4.0

### 1.2. Description
Please use the `lookahead/run_generate.py` file for running generation. We modify Huggingface's generation code to allow for the lookahead heuristics.

To generate with baseline decoding methods, i.e. the original generation methods, simply run the file without `--do_lookahead`. And to run the decoding with lookahead, simply add `--do_lookahead` and configure the appropriate configuration.

If using beam search with `--num_return_sequnces >1`, the output will be a json file where each item is a list of summary candidates. Otherwise, it is just a plain file where each line is the output summary.

### 1.3. Important arguments:
- `--model_name`: The Huggingface model to use for summarization
- `--document_file`: The input document, where each line represents a single document
- `--output_file`: The output summary file to write to.
- `--batch_size, --num_beams, --num_return_sequences, --max_input_length, --max_output_length, --do_sample`: The arguments used to control the base decoding method. Please refer to HuggingFace's original generation function for more details on this.
- `do_lookahead`: Controls whether to use the lookahead
- `--lookahead_length`: How many tokens to look into the future. By default, it should be the same as the `--max_output_length` to generate the full summary.
- `--lookhead_lambda`: The weight for the lookahead heuristics
- `--top_k`: How many top tokens the lookahead should consider to generate the full summary and provide the heuristics score. This should be set greater than the `lookhaed_beam`
- `--lookahead_decoding_type`: Which decoding strategy to use for the lookahead. The setup is similar to the base decoding strategy
- `--lookhead_beam`: The beam size of the lookahead used when `--lookhead_decoding_type=beam`
- `--scorer_model_type,--scorer_num_layers`:  The configuration for BERTScore scorer. Please refer to the official code for more details.

### 1.4. Examples
```
# greedy without lookhead
python run_generate.py --document_file xsum.document --output_file xsum_greedy.summary

# beam without lookahead
python run_generate.py --document_file xsum.document --output_file xsum_beam.json --num_beams 10 --num_return_sequences 10

# greedy with greedy lookahead
python run_generate.py --document_file xsum.document --output_file xsum_greedy_lookahead_greedy.summary --do_lookahead --lookahead_decoding_type greedy

```

## 2. Decoding with Ranking

### 2.1 Additional Dependencies
Please install the dependencies specified in 1.1 first.
- pandas

The scripts can be found under `ranking` directory. We assume that all files will be named with the same prefix, for example, `FILE_PREFIX=xsum_beam10` The steps to do the ranking:
1. Use the `lookahead/run_generate.py` file to generate beam outputs to generate a file `{FILE_PREFIX}_summary.json`
2. Generate the document file (with repeats according to the beam size) with `ranking/generate_documents.py xsum.document 10 ${FILE_PREFIX}.document`.
3. Please follow the instruction of the metrics' official repo to install and run the metrics. You may need to edit the code to allow for saving the scores. We expect each metric file to be in the json format. For example, `${FILE_PREFIX}_factcc.json, ${FILE_PREFIX}_dae.json, ...`
4. Run `ranking/rank.py --file_prefix ${FILE_PREFIX}`, and it will generate `${FILE_PREFIX}.csv`, which is the pandas csv file that stores all information, and `${FILE_PREFIX}_ranked_summary.txt`, that outputs the ranked summary with each line representing the ranked summary.

## 3. Training with Distillation

### 3.1. Additional Dependencies
Please install the dependencies specified in 1.1 first.
- nltk
- numpy
- [Deepspeed](https://github.com/microsoft/DeepSpeed) (Optional)

### 3.2. Description

The code can be found in `teacher-student/src`. The script `teacher-student/src/run_summarization.py` trains the summarization model with multiple references. The file is very similar to transformers' summarization code except for modifications to allow for multiple references. One additional argument is:
- `--additional_reference_file`: This should point to the file containing summaries of the training data, split into one line each.

An example of a training script can be found in `teacher-student/train_script.sh`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License Summary

The documentation is made available under the CC-BY-NC-4.0 License. See the LICENSE file.

## Citation

```
@inproceedings{wan-etal-2023-faithful-generation,
    title = "Faithfulness-Aware Decoding Strategies for Abstractive Summarization",
    author = "Wan, David  and
      Liu, Mengwen  and
      McKeown, Kathleen and
      Dreyer Markus and
      Bansal, Mohit",
      booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
      year={2023}
}
```