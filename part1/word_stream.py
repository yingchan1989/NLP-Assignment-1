def context_windows(words, C=5):
    '''A generator that yields context tuples of words, length C.
       Don't worry about emitting cases where we get too close to
       one end or the other of the array.

       Your code should be quite short and of the form:
       for ...:
         yield the_next_window
    '''
    # START YOUR CODE HERE
    tuples = []
    for i in range(len(words) - C + 1):
        tuples.append(words[i: i + C])

    return tuples
    # END YOUR CODE HERE


def cooccurrence_table(words, C=2):
    '''Generate cooccurrence table of words.
    Args:
       - words: a list of words
       - C: the # of words before and the number of words after
            to include when computing co-occurrence.
            Note: the total window size will therefore
            be 2 * C + 1.
    Returns:
       A list of tuples of (word, context_word, count).
       W1 occuring within the context of W2, d tokens away
       should contribute 1/d to the count of (W1, W2).
    '''
    table = []
    # START YOUR CODE HERE
    data = []
    dev_range = 2 * C + 1
    word_tuple = context_windows(words, dev_range)
    mid = dev_range / 2 + 1

    import itertools

    for j in range(0, len(word_tuple)):
        for i in range(- C, 0):
            distance = abs(i)
            data.append((word_tuple[j][mid - 1], word_tuple[j][mid + i - 1], 1 / float(distance)))

        for i in range(1, C + 1):
            distance = abs(i)
            data.append((word_tuple[j][mid - 1], word_tuple[j][mid + i - 1], 1 / float(distance)))

    keyfunc = lambda l: (l[0], l[1])
    data.sort(key=keyfunc)
    for key, rows in itertools.groupby(data, keyfunc):
        table.append((key[0], key[1], sum(r[2] for r in rows)))


    # END YOUR CODE HERE
    return table


def score_bigram(bigram, unigram_counts, bigram_counts, delta):
    '''Return the score of bigram.
    See Section 4 of Word2Vec (see notebook for link).

    Args:
      - bigram: the bigram to score: ('w1', 'w2')
      - unigram_counts: a map from word => count
      - bigram_counts: a map from ('w1', 'w2') => count
      - delta: the adjustment factor
    '''
    # START YOUR CODE HERE
    query = bigram

    if query[0] in unigram_counts:
        unigrams_a = unigram_counts[query[0]]
    else:
        unigrams_a = 0

    if query[1] in unigram_counts:
        unigrams_b = unigram_counts[query[1]]
    else:
        unigrams_b = 0

    if query in bigram_counts:
        bigram_count = bigram_counts[query]
        score = float(bigram_count - delta) / (unigrams_a * unigrams_b)
    else:
        score = float(0)

    return score
    # END YOUR CODE HERE
