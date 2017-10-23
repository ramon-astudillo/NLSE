import re
import twokenize


def max_reps(sentence, n=3):

    """
        Normalizes a string to at most n repetitions of the same character
        e.g, for n=3 and "helllloooooo" -> "helllooo"
    """
    new_sentence = ''
    last_c = ''
    max_counter = n
    for c in sentence:
        if c != last_c:
            new_sentence += c
            last_c = c
            max_counter = n
        else:
            if max_counter > 1:
                new_sentence += c
                max_counter -= 1
            else:
                pass
    return new_sentence


def preprocess(m):
    m = m.lower()
    m = max_reps(m)
    # replace user mentions with token '@user'
    user_regex = r".?@.+?( |$)|<@mention>"
    m = re.sub(user_regex, " @user ", m, flags=re.I)
    # replace urls with token 'url'
    m = re.sub(twokenize.url, " url ", m, flags=re.I)
    return twokenize.tokenize(m)


def tokenize_corpus(corpus):
    tokenized_corpus = []
    for tweet in corpus:
        tokenized_corpus.append([tweet[0], preprocess(' '.join(tweet[1]))])
    return tokenized_corpus
