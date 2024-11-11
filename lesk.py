from collections import Counter
from typing import *

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np
from numpy.linalg import norm

from ntlk import stop_tokenize
from wsd import evaluate, load_eval, load_word2vec, WSDToken


def mfs(sent: Sequence[WSDToken], w_index: int) -> Synset:
    """Most frequent sense of a word.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. See the WSDToken class in wsd.py
    for the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        w_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The most frequent sense for the given word.
    """
    # Get the lemma of the target word
    word = sent[w_index].lemma
    # Get the synsets for the word
    synsets = wn.synsets(word)
    # Return the most frequent sense (the first synset)
    if synsets:
        return synsets[0]
    else:
        return None


def lesk(sent: Sequence[WSDToken], w_index: int) -> Synset:
    """Simplified Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        w_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    word = sent[w_index].lemma
    context = [token.wordform for token in sent]
    # context_bag = Counter(stop_tokenize(' '.join(context)))
    context_bag = Counter(context)

    best_sense = None
    best_score = 0

    for sense in wn.synsets(word):
        # Build the signature from definition and examples
        signature_tokens = stop_tokenize(sense.definition())
        for example in sense.examples():
            signature_tokens.extend(stop_tokenize(example))
        signature_bag = Counter(signature_tokens)

        # Calculate overlap (intersection of the bags)
        overlap_count = sum((signature_bag & context_bag).values())

        if overlap_count > best_score:
            best_score = overlap_count
            best_sense = sense

    if best_sense:
        return best_sense
    else:
        # Fallback to most frequent sense if no overlap
        return mfs(sent, w_index)


def lesk_ext(sent: Sequence[WSDToken], w_index: int) -> Synset:
    """Extended Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        w_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    word = sent[w_index].lemma
    context = [token.wordform for token in sent]
    # context_bag = Counter(stop_tokenize(' '.join(context)))
    context_bag = Counter(context)

    best_sense = None
    best_score = 0

    for sense in wn.synsets(word):
        # Build the signature from definition, examples, hyponyms, holonyms, meronyms
        signature_tokens = stop_tokenize(sense.definition())
        for example in sense.examples():
            signature_tokens.extend(stop_tokenize(example))

        # Include hyponyms
        for related_sense in sense.hyponyms():
            signature_tokens.extend(stop_tokenize(related_sense.definition()))
            for example in related_sense.examples():
                signature_tokens.extend(stop_tokenize(example))

        # Include holonyms (member, part, substance)
        holonyms = sense.member_holonyms() + sense.part_holonyms() + sense.substance_holonyms()
        for holonym in holonyms:
            signature_tokens.extend(stop_tokenize(holonym.definition()))
            for example in holonym.examples():
                signature_tokens.extend(stop_tokenize(example))

        # Include meronyms (member, part, substance)
        meronyms = sense.member_meronyms() + sense.part_meronyms() + sense.substance_meronyms()
        for meronym in meronyms:
            signature_tokens.extend(stop_tokenize(meronym.definition()))
            for example in meronym.examples():
                signature_tokens.extend(stop_tokenize(example))

        signature_bag = Counter(signature_tokens)

        # Calculate overlap
        overlap_count = sum((signature_bag & context_bag).values())

        if overlap_count > best_score:
            best_score = overlap_count
            best_sense = sense

    if best_sense:
        return best_sense
    else:
        # Fallback to most frequent sense if no overlap
        return mfs(sent, w_index)


def lesk_cos(sent: Sequence[WSDToken], w_index: int) -> Synset:
    """Extended Lesk algorithm using cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        w_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    word = sent[w_index].lemma
    context = [token.wordform for token in sent]
    # context_tokens = stop_tokenize(' '.join(context))
    # context_bag = Counter(context_tokens)
    context_bag = Counter(context)

    best_sense = None
    best_score = -1

    # Build the vocabulary
    # vocab = set(context_tokens)
    vocab = set(context)

    # Collect signature_bags for all senses
    senses_signature_bags = []

    for sense in wn.synsets(word):
        signature_tokens = stop_tokenize(sense.definition())
        for example in sense.examples():
            signature_tokens.extend(stop_tokenize(example))

        # Include hyponyms, holonyms, meronyms as in lesk_ext
        related_senses = (sense.hyponyms() + sense.member_holonyms() + sense.part_holonyms() +
                          sense.substance_holonyms() + sense.member_meronyms() + sense.part_meronyms() +
                          sense.substance_meronyms())
        for related_sense in related_senses:
            signature_tokens.extend(stop_tokenize(related_sense.definition()))
            for example in related_sense.examples():
                signature_tokens.extend(stop_tokenize(example))

        signature_bag = Counter(signature_tokens)
        senses_signature_bags.append((sense, signature_bag))

        # Update vocabulary with signature_tokens
        vocab.update(signature_tokens)

    # Build vocab_index after collecting all signature_tokens
    vocab = list(vocab)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}

    # Convert context to vector
    context_vector = np.zeros(len(vocab))
    for word, count in context_bag.items():
        idx = vocab_index[word]
        context_vector[idx] = count

    for sense, signature_bag in senses_signature_bags:
        # Convert signature to vector
        signature_vector = np.zeros(len(vocab))
        for word, count in signature_bag.items():
            idx = vocab_index[word]
            signature_vector[idx] = count

        # Compute cosine similarity
        numerator = np.dot(context_vector, signature_vector)
        denominator = norm(context_vector) * norm(signature_vector)
        if denominator == 0:
            cos_sim = 0
        else:
            cos_sim = numerator / denominator

        if cos_sim > best_score:
            best_score = cos_sim
            best_sense = sense

    if best_sense:
        return best_sense
    else:
        return mfs(sent, w_index)


def lesk_cos_onesided(sent: Sequence[WSDToken], w_index: int) -> Synset:
    """Extended Lesk algorithm using one-sided cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        w_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """

    word = sent[w_index].lemma
    context = [token.wordform for token in sent]

    # Tokenize context tokens
    # context_tokens = stop_tokenize(' '.join(context))
    # context_bag = Counter(context_tokens)
    context_bag = Counter(context)

    # Build vocabulary from context tokens (original case)
    # vocab = set(context_tokens)
    vocab = set(context)
    vocab_index = {word: idx for idx, word in enumerate(sorted(vocab))}

    # Build a set of lowercased context tokens for comparison
    # context_tokens_lower = set(token.lower() for token in context_tokens)
    context_tokens_lower = set(token.lower() for token in context)

    # Convert context to vector
    context_vector = np.zeros(len(vocab))
    for word, count in context_bag.items():
        idx = vocab_index[word]
        context_vector[idx] = count

    best_sense = None
    best_score = -1

    for sense in wn.synsets(word):
        signature_tokens = stop_tokenize(sense.definition())
        for example in sense.examples():
            signature_tokens.extend(stop_tokenize(example))

        # Include related senses
        related_senses = (
                sense.hyponyms() + sense.member_holonyms() + sense.part_holonyms() +
                sense.substance_holonyms() + sense.member_meronyms() + sense.part_meronyms() +
                sense.substance_meronyms()
        )
        for related_sense in related_senses:
            signature_tokens.extend(stop_tokenize(related_sense.definition()))
            for example in related_sense.examples():
                signature_tokens.extend(stop_tokenize(example))

        # Filter signature tokens to include only those present in the context (case-insensitive)
        signature_tokens_filtered = [
            token for token in signature_tokens if token.lower() in context_tokens_lower
        ]

        if not signature_tokens_filtered:
            continue  # Skip senses with no overlapping words

        # Map signature tokens back to original case in vocab
        signature_bag = Counter()
        for token in signature_tokens_filtered:
            # Find the matching word in the context tokens
            # matching_words = [w for w in context_tokens if w.lower() == token.lower()]
            matching_words = [w for w in context if w.lower() == token.lower()]
            if matching_words:
                word_in_vocab = matching_words[0]
                signature_bag[word_in_vocab] += 1

        # Convert signature to vector
        signature_vector = np.zeros(len(vocab))
        for word, count in signature_bag.items():
            idx = vocab_index[word]
            signature_vector[idx] = count

        # Compute cosine similarity
        numerator = np.dot(context_vector, signature_vector)
        denominator = norm(context_vector) * norm(signature_vector)
        if denominator == 0:
            cos_sim = 0
        else:
            cos_sim = numerator / denominator

        if cos_sim > best_score:
            best_score = cos_sim
            best_sense = sense

    if best_sense:
        return best_sense
    else:
        return mfs(sent, w_index)

def lesk_w2v(sent: Sequence[WSDToken], w_index: int,
             vocab: Mapping[str, int], word2vec: np.ndarray) -> Synset:
    """Extended Lesk algorithm using word2vec-based cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    To look up the vector for a word, first you need to look up the word's
    index in the word2vec matrix, which you can then use to get the specific
    vector. More directly, you can look up a string s using word2vec[vocab[s]].

    To look up the vector for a *single word*, use the following rules:
    * If the word exists in the vocabulary, then return the corresponding
      vector.
    * Otherwise, if the lower-cased version of the word exists in the
      vocabulary, return the corresponding vector for the lower-cased version.
    * Otherwise, return a vector of all zeros. You'll need to ensure that
      this vector has the same dimensions as the word2vec vectors.

    But some wordforms are actually multi-word expressions and contain spaces.
    word2vec can handle multi-word expressions, but uses the underscore
    character to separate words rather than spaces. So, to look up a string
    that has a space in it, use the following rules:
    * If the string has a space in it, replace the space characters with
      underscore characters and then follow the above steps on the new string
      (i.e., try the string as-is, then the lower-cased version if that
      fails), but do not return the zero vector if the lookup fails.
    * If the version with underscores doesn't yield anything, split the
      string into multiple words according to the spaces and look each word
      up individually according to the rules in the above paragraph (i.e.,
      as-is, lower-cased, then zero). Take the mean of the vectors for each
      word and return that.
    Recursion will make for more compact code for these.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        w_index (int): The index of the target word in the sentence.
        vocab (dictionary mapping str to int): The word2vec vocabulary,
            mapping strings to their respective indices in the word2vec array.
        word2vec (np.ndarray): The word2vec word vectors, as a VxD matrix,
            where V is the vocabulary and D is the dimensionality of the word
            vectors.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    word = sent[w_index].lemma
    context_tokens = set(stop_tokenize(' '.join([token.wordform for token in sent])))

    # Get context vector (mean of word vectors)
    context_vectors = []
    for token in context_tokens:
        vec = get_word_vector(token, vocab, word2vec)
        if vec is not None:
            context_vectors.append(vec)
    if context_vectors:
        context_vector = np.mean(context_vectors, axis=0)
    else:
        context_vector = np.zeros(word2vec.shape[1])

    best_sense = None
    best_score = -1

    for sense in wn.synsets(word):
        signature_tokens = set(stop_tokenize(sense.definition()))
        for example in sense.examples():
            signature_tokens.update(stop_tokenize(example))

        # Include hyponyms, holonyms, meronyms as in lesk_ext
        for related_sense in sense.hyponyms() + sense.member_holonyms() + sense.part_holonyms() + sense.substance_holonyms() + sense.member_meronyms() + sense.part_meronyms() + sense.substance_meronyms():
            signature_tokens.update(stop_tokenize(related_sense.definition()))
            for example in related_sense.examples():
                signature_tokens.update(stop_tokenize(example))

        # Get signature vector (mean of word vectors)
        signature_vectors = []
        for token in signature_tokens:
            vec = get_word_vector(token, vocab, word2vec)
            if vec is not None:
                signature_vectors.append(vec)
        if signature_vectors:
            signature_vector = np.mean(signature_vectors, axis=0)
        else:
            signature_vector = np.zeros(word2vec.shape[1])

        # Compute cosine similarity
        numerator = np.dot(context_vector, signature_vector)
        denominator = norm(context_vector) * norm(signature_vector)
        if denominator == 0:
            cos_sim = 0
        else:
            cos_sim = numerator / denominator

        if cos_sim > best_score:
            best_score = cos_sim
            best_sense = sense

    if best_sense:
        return best_sense
    else:
        return mfs(sent, w_index)


def get_word_vector(word: str, vocab: Mapping[str, int], word2vec: np.ndarray) -> Optional[np.ndarray]:
    """Retrieve the word vector for a given word."""
    if word in vocab:
        return word2vec[vocab[word]]
    elif word.lower() in vocab:
        return word2vec[vocab[word.lower()]]
    elif ' ' in word:
        # Replace spaces with underscores
        word_underscore = word.replace(' ', '_')
        if word_underscore in vocab:
            return word2vec[vocab[word_underscore]]
        elif word_underscore.lower() in vocab:
            return word2vec[vocab[word_underscore.lower()]]
        else:
            # Split into words and get vectors
            tokens = word.split(' ')
            vectors = []
            for token in tokens:
                vec = get_word_vector(token, vocab, word2vec)
                if vec is not None:
                    vectors.append(vec)
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return None
    else:
        return None  # Return None if word not found


if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()
    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos, lesk_cos_onesided]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
