"""
Script used to calculate for the generated token counts of various datasets using the Mistral tokenizer.
"""

from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import Dataset, load_dataset
from typing import List, Dict, Callable, Any

import pandas

mistral_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('tokenizer/mistral')

datasets = {
    'capybara': load_dataset('LDJnr/Capybara', num_proc=8),
    'llama_instruct': load_dataset('togethercomputer/llama-instruct', num_proc=8),
    'meta_math': load_dataset('meta-math/MetaMathQA', num_proc=8),
    'slim_orca': load_dataset('Open-Orca/SlimOrca-Dedup', num_proc=8),
    'ultrachat_200k': load_dataset('HuggingFaceH4/ultrachat_200k', num_proc=8),
    'deita_10k': load_dataset('HuggingFaceH4/deita-10k-v0-sft', num_proc=8),
    'wizardlm_evol': load_dataset('Leon-Leee/Wizardlm_Evol_Instruct_v2_196K_backuped', num_proc=8)
}

def encode_capybara(batch: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """
    Encode the Capybara dataset by joining 'input' and 'output' fields of each message in the conversation.
    """
    joined_texts = [' '.join([message['input'] + ' ' + message['output'] for message in conversation]) for conversation in batch['conversation']]
    batch['token_counts'] = [len(mistral_tokenizer(text)['input_ids']) for text in joined_texts]
    return batch

def encode_llama_instruct(batch: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Encode the Llama Instruct dataset by tokenizing the 'text' field.
    """
    batch['token_counts'] = [len(mistral_tokenizer(text)['input_ids']) for text in batch['text']]
    return batch

def encode_meta_math(batch: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Encode the Meta Math dataset by combining 'query' and 'response' fields.
    """
    combined_texts = [query + ' ' + response for query, response in zip(batch['query'], batch['response'])]
    batch['token_counts'] = [len(mistral_tokenizer(text)['input_ids']) for text in combined_texts]
    return batch

def encode_conversations(batch: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """
    Encode datasets with conversation format by joining 'value' fields of each message.
    """
    joined_texts = [' '.join([message['value'] for message in conversation]) for conversation in batch['conversations']]
    batch['token_counts'] = [len(mistral_tokenizer(text)['input_ids']) for text in joined_texts]
    return batch

def encode_messages(batch: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """
    Encode datasets with message format by joining 'content' fields of each message.
    """
    joined_texts = [' '.join([message['content'] for message in messages]) for messages in batch['messages']]
    batch['token_counts'] = [len(mistral_tokenizer(text)['input_ids']) for text in joined_texts]
    return batch

def process_dataset(name: str, dataset: Dataset, encode_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> int:
    """
    Process the dataset using the provided encoding function and return the total token count.
    
    Args:
        name (str): The name of the dataset.
        dataset (Dataset): The dataset to be processed.
        encode_fn (Callable): The encoding function to apply to the dataset.
        
    Returns:
        int: The total token count for the dataset.
    """
    processed_data = dataset.map(encode_fn, batched=True, num_proc=4)
    dataframe: pandas.DataFrame = processed_data.to_pandas()
    token_count_sum = dataframe['token_counts'].sum()
    print(f'{name}: {token_count_sum}')
    return token_count_sum

if __name__ == '__main__':
    total = 0
    total += process_dataset('LDJnr/Capybara', datasets['capybara']['train'], encode_capybara)
    total += process_dataset('togethercomputer/llama-instruct', datasets['llama_instruct']['train'], encode_llama_instruct)
    total += process_dataset('meta-math/MetaMathQA', datasets['meta_math']['train'], encode_meta_math)
    total += process_dataset('Open-Orca/SlimOrca-Dedup', datasets['slim_orca']['train'], encode_conversations)
    total += process_dataset('HuggingFaceH4/ultrachat_200k', datasets['ultrachat_200k']['train_sft'], encode_messages)
    total += process_dataset('HuggingFaceH4/deita-10k-v0-sft', datasets['deita_10k']['train_sft'], encode_messages)
    total += process_dataset('Leon-Leee/Wizardlm_Evol_Instruct_v2_196K_backuped', datasets['wizardlm_evol']['train'], encode_conversations)
    print('total tokens:', total)
