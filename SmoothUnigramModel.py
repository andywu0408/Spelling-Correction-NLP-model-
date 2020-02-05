import math, collections


class SmoothUnigramModel:
    """Smooth Unigram Model"""

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        # TODO your code here
        # Tip: To get words from the corpus, try
        #    for sentence in corpus.corpus:
        #       for datum in sentence.data:
        #         word = datum.word
        for sentence in corpus.corpus:
            for datum in sentence.data:
                token = datum.word
                self.unigramCounts[token] = self.unigramCounts[token] + 1
                self.total += 1

        self.unigramCounts['UNK'] = 0
        self.total += 1

        for entry in self.unigramCounts:
            # increment every word count by one (Laplace smoothing)
            self.unigramCounts[entry] = self.unigramCounts[entry] + 1
            # normalize the total by adding one every time we smooth the count.
            self.total += 1

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        for token in sentence:
            count = self.unigramCounts[token]
            if count <= 0:
                count = self.unigramCounts['UNK']
            if count > 0:
                score += math.log(count)
                score -= math.log(self.total)

        return score
