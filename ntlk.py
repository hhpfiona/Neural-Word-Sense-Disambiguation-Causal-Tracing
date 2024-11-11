import typing as T
from string import punctuation

import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt_tab')

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize


def deepest():
    """Find and print the synset with the largest maximum depth along with its
    depth on each of its hyperonym paths.

    Returns:
        None
    """
    max_depth = 0
    deepest_synset = None
    depths = []
    for s in wn.all_synsets():
        hypernym_paths = s.hypernym_paths()
        s_max_depth = max(len(path) - 1 for path in hypernym_paths)
        if s_max_depth > max_depth:
            max_depth = s_max_depth
            deepest_synset = s
            depths = [len(path) - 1 for path in hypernym_paths]
    # After processing all synsets
    print(f'Deepest synset: {deepest_synset}')
    print(f'Depths on each path to a root hypernym: {depths}')


def superdefn(synset: str) -> T.List[str]:
    """Get the "superdefinition" of a synset. (Yes, superdefinition is a
    made-up word. All words are made up.)

    We define the superdefinition of a synset to be the list of word tokens,
    here as produced by word_tokenize, in the definitions of the synset, its
    hyperonyms, and its hyponyms.

    Args:
        synset (str): The name of the synset to look up

    Returns:
        list of str: The list of word tokens in the superdefinition of s

    Examples:
        >>> superdefn('toughen.v.01')
        ['make', 'tough', 'or', 'tougher', 'gain', 'strength', 'make', 'fit']
    """
    s = wn.synset(synset)
    definitions = []
    definitions.append(s.definition())
    hypernyms = s.hypernyms()
    hyponyms = s.hyponyms()
    for hypernym in hypernyms:
        definitions.append(hypernym.definition())
    for hyponym in hyponyms:
        definitions.append(hyponym.definition())
    tokens = []
    for definition in definitions:
        tokens.extend(word_tokenize(definition))
    return tokens


def stop_tokenize(s: str) -> T.List[str]:
    """Word-tokenize and remove stop words and punctuation-only tokens.

    Args:
        s (str): String to tokenize

    Returns:
        list[str]: The non-stopword, non-punctuation tokens in s

    Examples:
        >>> stop_tokenize('The Dance of Eternity, sir!')
        ['Dance', 'Eternity', 'sir']
    """
    tokens = word_tokenize(s)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for token in tokens:
        if token.lower() in stop_words:
            continue
        if all(c in punctuation for c in token):
            continue
        filtered_tokens.append(token)
    return filtered_tokens


if __name__ == '__main__':
    import doctest
    doctest.testmod()