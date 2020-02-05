import math, collections


class SmoothBigramModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        # TODO your code here
        # Tip: To get words from the corpus, try
        for sentence in corpus.corpus:
            for i in range(len(sentence.data) - 1):
                token = '%s %s' % (sentence.data[i].word, sentence.data[i + 1].word)

                self.bigramCounts[token] = self.bigramCounts[token] + 1
                self.total += 1

        self.bigramCounts['UNK'] = 0

        for entry in self.bigramCounts:
            # increment every word count by one (Laplace smoothing)
            self.bigramCounts[entry] = self.bigramCounts[entry] + 1
            # normalize the total by adding one every time we smooth the count.
            self.total += 1

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here
        score = 0.0

        for i in range(0, len(sentence) - 1):
            token = '%s %s' % (sentence[i], sentence[i + 1])
            if self.bigramCounts[token] <= 0:
                count = self.bigramCounts['UNK']
            else:
                count = self.bigramCounts[token]

            score += math.log(count)
            score -= math.log(self.total)

        return score
