import json
from datasets import load_dataset


def create_test_jsonl(huggingface_dataset_id: str, split: str, number_of_rows=16):
    dataset_stream = load_dataset(huggingface_dataset_id, split=split, streaming=True)

    sampled_rows = []

    for i, row in enumerate(dataset_stream):
        sampled_rows.append(row)

        if len(sampled_rows) == number_of_rows:
            break

    filename = huggingface_dataset_id.split('/')[1]

    with open(f'{filename.lower()}_test.jsonl', 'w') as outfile:
        for row in sampled_rows:
            json.dump(row, outfile)  # Dump each row as a JSON object
            outfile.write('\n')


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(create_test_jsonl, as_positional=False)
