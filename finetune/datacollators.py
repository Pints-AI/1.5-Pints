from dataclasses import dataclass
from typing import Sequence, Dict, List
from copy import deepcopy
from transformers import PreTrainedTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

# IGNORE_INDEX of -100 will tell pytorch to skip loss calculation
# It is also the default pytorch ignore_index.
# See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
IGNORE_INDEX = -100

@dataclass
class DataCollatorForCausalLMDecoderOnlyChatML(object):

    '''
    This DataCollator is purpose-made for ChatML templates.
    Referenced: https://medium.com/@xuebinbin12/fine-tuning-chat-based-llm-with-multi-turn-conversational-data-part-i-d8c64d01a20d
    '''
    tokenizer: PreTrainedTokenizer
    target_max_len: int

    # Decoder only doesn't require source source_max_length
    # source_max_length: int

    # `train_on_source` in this context refers to calculating prediction losses on the "user" prompts
    # In some cases where there are things to learn from the user prompts, this might help.
    # However in some finetuning dataset, the user prompts mimick a real-world poor quality prompts.
    # And usually doesn't contain anything useful. Therefore, prediction losses should be ignored,
    # So that the model doesn't learn to mimick it.
    train_on_source: bool  
    add_special_tokens: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        if self.train_on_source:
            message = 'In some cases where there are things to learn from the user prompts, this might help.'
            message += 'However in some finetuning dataset, the user prompts mimick a real-world poor quality prompts.'
            message += 'And usually doesn\'t contain anything useful. Therefore, prediction losses should be ignored.'
            message += 'so that the model doesn\'t learn to mimick it.'
            match input(f'{message}\n\nAre you sure you want to continue? (yes/no): '):
                case 'yes':
                    pass
                case _:
                    exit()

        # In decoder-only model, we concat source and targets
        contents: List[str] = []

        for example in instances:
            content = example['input'] + example['output']
            contents.append(content)

        # Default is to not train on source, so we ignore losses on the source with IGNORE_INDEX
        if not self.train_on_source:

            # identify the start and end of the sequence to ignore losses
            # <|im_start|>user\nhello~<|im_end|>\n<|im_start|>assistant\nHow may I help?<|im_end|>
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # The ^ represents the positions that we will apply IGNORE_INDEX

            # To do the ignore_index for cross-entropy loss, we need the start tokens and the end tokens to ignore
            start_tokens = self.tokenizer.encode(
                '<|im_start|>user\n',
                max_length=999,
                truncation=True,
                add_special_tokens=False,
            )
            end_tokens = self.tokenizer.encode(
                '<|im_start|>assistant\n',
                max_length=999,
                truncation=True,
                add_special_tokens=False,
            )

            tokenized_contents = self.tokenizer(
                contents,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=self.add_special_tokens,

                # don't pad here, apply padding later
                # so that padding can be applied batch-wise to pad to max length of the batch
                # saving memory
                # padding='max_length',

                # need this to allow grouping if required
                # see https://adityaroc.medium.com/bucketing-a-technique-to-reduce-train-time-complexity-for-seq2seq-model-84774a34b6a0
                return_length=True
            )

            input_ids = []
            labels = []

            for tokenized_content in tokenized_contents['input_ids']:

                # Append the tokenized_content as input_ids
                # And make a copy to turn it into labels with IGNORE_INDEX applied
                input_ids.append(torch.tensor(tokenized_content))
                copied_tokenized_content = tokenized_content.copy()

                # Loop through the whole content to find the first index of the start sequence
                # - len(start_tokens) + 1 helps to exit slightly earlier, as remain sequence isn't long enough to contain the `start_tokens`
                current_index = 0
                while current_index < (len(copied_tokenized_content) - len(start_tokens) + 1):

                    # find the first index of the start sequence
                    if self.find_subsequence(copied_tokenized_content, start_tokens, current_index):

                        # So the current_index is the starting point to apply the IGNORE_INDEX
                        start_index = current_index

                        # We start seeking for the end index, and we advance the positions by the number of start_tokens
                        end_index = current_index + len(start_tokens)

                        # In case when we enter this condition, and no end sequence is found,
                        # we need this flag to break the outer loop
                        found_end = False

                        # Loop through the content starting from the token right after the `start_tokens` sequence.
                        # Same thing, - len(end_tokens) + 1 helps to exit slightly earlier
                        while end_index < (len(copied_tokenized_content) - len(end_tokens) + 1):
                        
                            if self.find_subsequence(copied_tokenized_content, end_tokens, end_index):

                                # The end_index is found, we need to advance it by the number of end_tokens which will also be ignored.
                                # GOTCHA: There is no need to do: + len(end_tokens) + 1
                                #
                                # EXPLANATION:
                                #
                                # end_index is the start of the `end_tokens` seqeunce.
                                # E.g if `end_tokens` is       [32002, 13]
                                # And the sequence is [0, 0, 0, 32002, 13]
                                # Positions:           0  1  2      3   4
                                # Currently, end_index would be 3
                                # And the end_slice_index should be 5, which is therefore 3 + len(end_tokens)
                                end_slice_index = end_index + len(end_tokens)
                                end_index = end_index + len(end_tokens)

                                # Apply the IGNORE_INDEX
                                # Up to, but not including `end_slice_index`
                                copied_tokenized_content[start_index:end_slice_index] = [IGNORE_INDEX] * (end_slice_index - start_index)

                                
                                # The end_slice_index is the position after the one that is ignored.
                                # Remember, ealier we applied IGNORE_INDEX to up to but not including `end_slice_index`
                                current_index = end_slice_index
                                found_end = True

                                # Break out of this loop as it's complete.
                                # To start the process again to find the start tokens
                                break

                            else: 

                                # No end sequence is found in this position, search the next position
                                end_index += 1

                        # In case there was a start sequence found, but the end sequence was truncated,
                        # found_end is False, we break the outer loop
                        if found_end is False:
                            break

                    else:
                        current_index += 1

                # Because the first index is the start token
                # In the case where start token is different from <|im_start|>, it should also be ignored
                copied_tokenized_content[0] = IGNORE_INDEX
                labels.append(torch.tensor(copied_tokenized_content))

            # Apply padding here instead to save memory instead of padding using the tokenizer
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            data_dict = {
                'input_ids': input_ids,
                # mask out the attention for pad tokens to ignore attention
                'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
                'labels': labels
            }
            return data_dict

        else:
            
            # tokenize everything and return it

            tokenized_contents = self.tokenizer(
                contents,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=True,

                # don't pad here
                # padding='max_length',

                # need this to allow grouping if required
                # see https://adityaroc.medium.com/bucketing-a-technique-to-reduce-train-time-complexity-for-seq2seq-model-84774a34b6a0
                return_length=True
            )

            input_ids = torch.tensor(tokenized_contents['input_ids'])
            labels = deepcopy(input_ids)

            # Apply padding here instead to save memory instead of padding using the tokenizer
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

            data_dict = {
                'input_ids': input_ids,
                # mask out the attention for pad tokens to ignore attention
                'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
                'labels': labels
            }

            return data_dict
        

    def find_subsequence(self, full_sequence: List[int], subsequence: List[int], start_from_index = 0):

        '''
        Check if sequence exist.
        '''

        # Suppose full sequence is [1, 99, 77, 66, 99, 88]
        # Lets say the subsequnce is [77, 66, 99]
        # We can return true when the `start_from_index is 2:`
        #
        #              [1, 99, 77, 66, 99, 88]
        # Positions:    0   1   2   3   4   5 
        # Subsequence:        [77, 66, 99]
        #
        # The correct slice to compare from the full sequence is position 2, to position 4.
        # Which should be [2:5] (up to but not including 5)
        #
        # Since the length of the subseqence is 3, the `end_index` can be `start_from_index + 3 = 5` 
        end_index = start_from_index + len(subsequence)

        if full_sequence[start_from_index:end_index] == subsequence:
            return True
        else:
            return False

 
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=True,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            
            # If `predict_with_generate` is not activated
            # We will concat the source and target
            # Otherwise, only the source is used.
            if not self.predict_with_generate:
                # Contat source and target
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))

                # If not training on source, we mask source in the labels with with IGNORE_INDEX
                # NOTES:
                # In encoder-decoder, default is to train on target (output) only
                # In some cases, training on input, where inputs contain quality information,
                # may produce better results.
                # See https://github.com/artidoro/qlora/issues/188
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }

        if labels is not None:
            data_dict['labels'] = labels

        return data_dict