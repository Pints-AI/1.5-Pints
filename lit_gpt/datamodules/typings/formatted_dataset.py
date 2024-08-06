from typing import List, TypedDict


class FormattedSFTDict(TypedDict):
    instruction: str
    input: str
    output: str


# In a single turn dataset, each instruction, input, and output tuples
# is a complete conversation discussing the same subject.
FormattedSFTSingleturnConversation = FormattedSFTDict
FormattedSFTSingleturnDataset = List[FormattedSFTDict]

# A `Conversation` comprise multiple instruction, input, and output tuples.
# Everything inside of one `Converstion` is discussing the same subject
FormattedSFTMultiturnConversation = List[FormattedSFTDict]
# Therefore, a multiturn dataset will comprise multiple `Conversations`
# each discussing a different subject.
FormattedSFTMultiturnDataset = List[List[FormattedSFTDict]]
