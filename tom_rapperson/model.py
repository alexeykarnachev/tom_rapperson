from transformers import GPT2LMHeadModel


def get_model_from_huggingface_pretrained(model_name, vocab_size) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(model_name)
    _resize_embeddings(model, vocab_size)
    return model


def _resize_embeddings(model: GPT2LMHeadModel, vocab_size):
    old_size = model.base_model.wte.num_embeddings
    n_new = vocab_size - old_size
    if n_new < 0:
        raise ValueError(f"Can't resize embeddings: new vocab size ({vocab_size}) can not be less than the "
                         f"old embeddings number ({old_size}).")
    model.resize_token_embeddings(vocab_size)
    idx = vocab_size - n_new
    reference_emb = model.base_model.wte.weight.data.mean(0)
    model.base_model.wte.weight.data[idx:] = reference_emb.unsqueeze(0)
