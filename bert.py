import typing as T
from collections import defaultdict

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import torch
from torch import Tensor
from torch.linalg import norm

from tqdm import tqdm, trange

from lesk import mfs
from wsd import (batch_evaluate, load_bert, run_bert, load_eval, load_train,
                 WSDToken)


def gather_sense_vectors(corpus: T.List[T.List[WSDToken]],
                         bs: int = 32) -> T.Dict[str, Tensor]:
    """Gather sense vectors using BERT run over a corpus.

    It is much more efficient to batch the sentences up than it is
    to do one sentence at a time, and you can further improve (~twice as fast)
    if you sort the corpus by sentence length first.

    The procedure for this function is as follows:
    * Use run_bert to run BERT on each batch
    * Go through all of the WSDTokens in the input batch. For each one, if the
      token has any synsets assigned to it (check WSDToken.synsets), then add
      the BERT output vector to a list of vectors for that sense (**not** for
      the token!).
    * Once this is done for all batches, then for each synset that was seen
      in the corpus, compute the mean of all vectors stored in its list.
    * That yields a single vector associated to each synset; return this as
      a dictionary.

    The run_bert function will handle tokenizing the batch for BERT, including
    padding the tokenized sentences so that each one has the same length, as
    well as converting it to a PyTorch tensor that lives on the GPU. It then
    runs BERT on it and returns the output vectors from the top layer.

    An important point: the tokenizer will produce more tokens than in the
    original input, because sometimes it will split one word into multiple
    pieces. BERT will then produce one vector per token. In order to
    produce a single vector for each *original* word token, to
    then use that vector for its various synsets, need to align the
    output tokens back to the originals. Sometimes there are multiple
    vectors for a single token in the input data; take the mean of these to
    yield a single vector per token. This vector can then be used like any
    other in the procedure described above.

    To provide the needed information to compute the token-word alignments,
    run_bert returns an offset mapping. For each token, the offset mapping
    provides substring indices, indicating the position of the token in the
    original word (or [0, 0] if the token doesn't correspond to any word in the
    original input, such as the [CLS], [SEP], and [PAD] tokens). You can
    inspect the returned values from run-bert in a debugger and/or try running
    the tokenizer on your own test inputs. Below are a couple examples.

        >>> from wsd import load_bert
        >>> load_bert()
        >>> from wsd import TOKENIZER as tknz
        >>> tknz('This is definitely a sentence.')
        {'input_ids': [101, 1188, 1110, 5397, 170, 5650, 119, 102],
         'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],
         'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
        >>> out = tknz([['Multiple', ',', 'pre-tokenized', 'sentences', '!'], \
                        ['Much', 'wow', '!']], is_split_into_words=True, \
                        padding=True, return_offsets_mapping=True)
        >>> out.tokens(0)
        ['[CLS]', 'Multiple', ',', 'pre', '-', 'token', '##ized', 'sentences',
         '!', '[SEP]']
        >>> out.tokens(1)
        ['[CLS]', 'Much', 'w', '##ow', '!', '[SEP]', '[PAD]', '[PAD]', '[PAD]',
        '[PAD]']
        >>> out['offset_mapping']
        [[[0, 0], [0, 8], [0, 1], [0, 3], [3, 4], [4, 9], [9, 13], [0, 9],
         [0, 1], [0, 0]],
         [[0, 0], [0, 4], [0, 1], [1, 3], [0, 1], [0, 0], [0, 0], [0, 0],
         [0, 0], [0, 0]]]

    Args:
        corpus (list of list of WSDToken): The corpus to use.
        bs (int): The batch size to use.

    Returns:
        dictionary mapping synsets IDs to Tensor: A dictionary that can be used to
        retrieve the (PyTorch) vector for a given sense.
    """
    sense_vectors = defaultdict(list)
    corpus = sorted(corpus, key=len)
    for batch_n in trange(0, len(corpus), bs, desc='gathering',
                          leave=False):
    
        batch = corpus[batch_n:batch_n + bs]

        # Extract wordforms from each sentence in the batch
        sentences = [[token.wordform for token in sentence] for sentence in batch]

        # Run BERT on the batch to get output vectors and offset mappings
        bert_outputs, offset_mappings = run_bert(sentences)

        # Process each sentence in the batch
        for sentence_idx, sentence_tokens in enumerate(batch):
            bert_output = bert_outputs[sentence_idx]
            offsets = offset_mappings[sentence_idx]

            # Map each original word to its corresponding token indices
            word_to_token_indices = defaultdict(list)
            current_word_index = 0

            for token_index, (start_offset, end_offset) in enumerate(offsets):
                # Skip special tokens ([CLS], [SEP], [PAD])
                if start_offset == 0 and end_offset == 0:
                    continue

                # Start a new word when encountering a token with a new start offset
                if start_offset == 0:
                    current_word_index += 1

                word_to_token_indices[current_word_index].append(token_index)

            # Compute vectors for each word and associate them with synsets
            for word_index, token_indices in word_to_token_indices.items():
                # Collect BERT vectors for the tokens of the current word
                token_vectors = [bert_output[token_idx] for token_idx in token_indices]

                # Compute the mean vector if there are multiple token vectors
                if len(token_vectors) > 1:
                    word_vector = torch.mean(torch.stack(token_vectors), dim=0)
                else:
                    word_vector = token_vectors[0]

                # Retrieve synsets for the current word token
                synsets = sentence_tokens[word_index - 1].synsets
                if not synsets:
                    continue

                # Store the word vector for each associated synset
                for synset in synsets:
                    sense_vectors[synset].append(word_vector)

    # Compute the mean vector for each synset
    synset_mean_vectors = {
        synset: torch.stack(vectors).mean(dim=0)
        for synset, vectors in sense_vectors.items() if vectors
    }

    return synset_mean_vectors

def bert_1nn(batch: T.List[T.List[WSDToken]],
             indices: T.Iterable[T.Iterable[int]],
             sense_vectors: T.Mapping[str, Tensor]) -> T.List[T.List[Synset]]:
    """Find the best sense for specified words in a batch of sentences using
    the most cosine-similar sense vector.

    See the docstring for gather_sense_vectors above for examples of how to use
    BERT. After running BERT on the input batch and associating a single
    vector for each input token in the same way, you can
    compare the vector for the target word with the sense vectors for its
    possible senses, and then return the sense with the highest cosine
    similarity.

    In case none of the senses have vectors, return the most frequent sense
    (e.g., by just calling mfs()).

    **IMPORTANT**: When computing the cosine similarities and finding the sense
    vector with the highest one for a given target word, do not use any loops.
    Implement this aspect via matrix-vector multiplication and other PyTorch
    ops.

    Args:
        batch (list of list of WSDToken): The batch of sentences containing
            words to be disambiguated.
        indices (list of list of int): The indices of the target words in the
            batch sentences.
        sense_vectors: A dictionary mapping synset IDs to PyTorch vectors, as
            generated by gather_sense_vectors(...).

    Returns:
        pred: The predictions of the correct sense for the given words.
    """
    predictions = []

    # Sort the batch and corresponding indices by sentence length for efficient BERT processing
    sorted_batch = sorted(batch, key=len)
    sorted_order = sorted(range(len(batch)), key=lambda i: len(batch[i]))
    sentences = [[token.wordform for token in sentence] for sentence in sorted_batch]

    # Run BERT to get output vectors and offset mappings
    bert_outputs, offset_mappings = run_bert(sentences)

    # Preprocess sense vectors for efficient similarity computation
    synset_ids = list(sense_vectors.keys())
    synset_vectors = torch.stack([sense_vectors[synset_id] for synset_id in synset_ids])
    normalized_synset_vectors = synset_vectors / synset_vectors.norm(dim=1, keepdim=True)

    # Process each sentence in the sorted batch
    for sorted_idx, sentence_tokens in enumerate(sorted_batch):
        sentence_predictions = []
        bert_output = bert_outputs[sorted_idx]
        offsets = offset_mappings[sorted_idx]

        # Map original words to corresponding BERT token indices
        word_to_tokens = defaultdict(list)
        current_word = -1
        for token_idx, (start, end) in enumerate(offsets.tolist()):
            if start == 0 and end == 0:
                continue  # Skip special tokens
            if start == 0:
                current_word += 1  # New word starts
            word_to_tokens[current_word].append(token_idx)

        # Process each target word for disambiguation
        original_idx = sorted_order.index(sorted_idx)
        for word_idx in indices[original_idx]:
            if word_idx < 0 or word_idx >= len(sentence_tokens):
                raise IndexError(f"Word index {word_idx} out of range for sentence length {len(sentence_tokens)}.")

            token_indices = word_to_tokens.get(word_idx, [])
            if not token_indices:
                sentence_predictions.append(mfs(sentence_tokens, word_idx))
                continue

            # Compute mean vector for the target word
            word_vectors = bert_output[token_indices].mean(dim=0)
            normalized_word_vector = word_vectors / word_vectors.norm()

            # Retrieve possible synsets for the word lemma
            lemma = sentence_tokens[word_idx].lemma
            candidate_synsets = wn.synsets(lemma)
            if not candidate_synsets:
                sentence_predictions.append(mfs(sentence_tokens, word_idx))
                continue

            # Filter synsets with available vectors
            valid_synsets, valid_indices = [], []
            for synset in candidate_synsets:
                synset_id = synset.name()
                if synset_id in sense_vectors:
                    valid_synsets.append(synset)
                    valid_indices.append(synset_ids.index(synset_id))

            if not valid_synsets:
                sentence_predictions.append(mfs(sentence_tokens, word_idx))
                continue

            # Compute cosine similarity and select the best synset
            valid_vectors = normalized_synset_vectors[valid_indices]
            similarity_scores = torch.matmul(valid_vectors, normalized_word_vector)
            best_synset = valid_synsets[torch.argmax(similarity_scores).item()]

            sentence_predictions.append(best_synset)

        predictions.append(sentence_predictions)

    # Reorder predictions to match the original batch order
    reordered_predictions = [None] * len(batch)
    for sorted_idx, original_idx in enumerate(sorted_order):
        reordered_predictions[original_idx] = predictions[sorted_idx]

    return reordered_predictions

if __name__ == '__main__':
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        tqdm.write(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        tqdm.write('Running on CPU.')

    with torch.no_grad():
        load_bert()
        train_data = load_train()
        eval_data = load_eval()

        sense_vecs = gather_sense_vectors(train_data)
        batch_evaluate(eval_data, bert_1nn, sense_vecs)
