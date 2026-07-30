from transformers import BertTokenizer, CLIPTokenizer


def get_bert_tokenizer(model_name="bert-base-uncased"):
    """
    BERT tokenizer for BERT and UNITER text streams
    """
    return BertTokenizer.from_pretrained(model_name)


def get_clip_tokenizer(model_name="openai/clip-vit-base-patch32"):
    """
    CLIP tokenizer for CLIP text encoder
    """
    return CLIPTokenizer.from_pretrained(model_name)
