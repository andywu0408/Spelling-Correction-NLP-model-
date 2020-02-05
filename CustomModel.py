import math, collections


class CustomModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 0.0)
        self.bigramCounts = collections.defaultdict(lambda: 0.0)
        self.total = 0  # for unsmoothed bigram
        self.smoothedTotal = 0  # for smoothed unigram
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        # train unsmoothed bigrams
        for sentence in corpus.corpus:
            for i in range(len(sentence.data) - 1):
                token = '%s %s' % (sentence.data[i].word, sentence.data[i + 1].word)

                self.bigramCounts[token] = self.bigramCounts[token] + 1
                self.total += 1

        # train unigram
        for sentence in corpus.corpus:
            for datum in sentence.data:
                token = datum.word
                self.unigramCounts[token] = self.unigramCounts[token] + 1
                self.smoothedTotal += 1

        self.unigramCounts['UNK'] = 0
        self.smoothedTotal += 0.2

        # smooth unigram
        for entry in self.unigramCounts:
            # increment every word count by one (Laplace smoothing)
            self.unigramCounts[entry] = self.unigramCounts[entry] + 0.2
            # normalize the total by adding one every time we smooth the count.
            self.smoothedTotal += 0.2

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0

        for i in range(0, len(sentence) - 1):
            token = '%s %s' % (sentence[i], sentence[i + 1])

            # Never saw this word, use smoothed unigram model

            if self.bigramCounts[token] <= 0:

                count = self.unigramCounts[sentence[i + 1]]
                if count <= 0:
                    count = self.unigramCounts['UNK']
                score += math.log(count)
                score -= math.log(self.smoothedTotal)

            # Have seen this word, Use unsmoothed bigram model
            else:
                count = self.bigramCounts[token]

            score += math.log(count)
            score -= math.log(self.total)

        return score
