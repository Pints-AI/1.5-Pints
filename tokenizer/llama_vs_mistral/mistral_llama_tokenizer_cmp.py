from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import IterableDataset, load_dataset
from typing import Tuple

import multiprocessing
import time

# Constants
NUM_CPU = multiprocessing.cpu_count() - 2 # Use available CPUs minus two for processing
TOTAL_ROW = 35000000 # Total number of rows to process
BATCH_SIZE = TOTAL_ROW // NUM_CPU # Calculate batch size based on number of CPUs

# Load dataset with streaming mode
falcon_dataset = load_dataset("tiiuae/falcon-refinedweb", split='train', streaming=True)

# Initialize tokenizers
mistral_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('tokenizer/mistral')
llama_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('tokenizer/llama2')

def encode(iterable_dataset: IterableDataset) -> Tuple[int, int]:
    """Encodes data using Mistral and Llama tokenizers."""
    mapped_dataset = iterable_dataset.map(lambda data: {
        'token_count_mistral': len(mistral_tokenizer(data['content'])['input_ids']),
        'token_count_llama': len(llama_tokenizer(data['content'])['input_ids']),
        'word_count': len(data['content'].split(' ')),
        'character_count': len(data['content'])
    })

    # Iterate and sum the token counts for the current batch
    mistral_tokens_sum, llama_tokens_sum = 0, 0
    word_count_sum, character_count_sum = 0, 0
    for row in mapped_dataset:
        mistral_tokens_sum += row['token_count_mistral']
        llama_tokens_sum += row['token_count_llama']
        word_count_sum += row['word_count']
        character_count_sum += row['character_count']

    print("One batch done!")
    return mistral_tokens_sum, llama_tokens_sum, word_count_sum, character_count_sum

if __name__ == '__main__':
    start_time = time.time()

    # Split the dataset into chunks for each CPU
    batch_data = [falcon_dataset.skip(batch_count * BATCH_SIZE).take(BATCH_SIZE) for batch_count in range(NUM_CPU)]

    # Process batches in parallel
    total_mistral_tokens, total_llama_tokens = 0, 0
    total_word_counts, total_character_counts = 0, 0
    with multiprocessing.Pool(processes=NUM_CPU) as pool:
        aggregated_counts = pool.map(encode, batch_data)
        for mistral_tokens, llama_tokens, word_counts, character_counts in aggregated_counts:
            total_mistral_tokens += mistral_tokens
            total_llama_tokens += llama_tokens
            total_word_counts += word_counts
            total_character_counts += character_counts

    print(f"Total tokens used for Mistral Tokenizer: {total_mistral_tokens}")
    print(f"Total tokens used for Llama Tokenizer: {total_llama_tokens}")
    print(f"Ratio: {total_mistral_tokens / total_llama_tokens}\n")

    print(f"Total Words tokenized: {total_word_counts}")
    print(f"Total Characters tokenized: {total_character_counts}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Process took {int(hours)}h {int(minutes)}m {seconds:.4f}s to complete.")
