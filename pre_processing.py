from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize(data):
    '''
    tokenize sentences and transfer them into word IDs
    '''
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    ids = []

    # For every sentence
    for sent in data:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens = True
        )
    
        ids.append(encoded_sent)
    
    return ids


def pad(ids):
    '''
    pad/truncating all the sentences to 64 values
    '''
    MAX_LEN = 64

    ids = pad_sequences(
        ids, 
        maxlen=MAX_LEN,
        dtype = "long",
        value = 0,
        truncating="post",
        padding="post"
        )
    
    return ids


def create_masks(ids):
    '''
    create attention masks
    '''
    attention_masks = []

    for sent in ids:
        att_mask = [int(token_id>0) for token_id in sent]

        attention_masks.append(att_mask)
    
    return attention_masks




