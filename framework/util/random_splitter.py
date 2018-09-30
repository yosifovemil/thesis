# This class splits a dataset in a random way using percentages allocated for
# each part. It is used when training a model and needing a training dataset
# and test dataset.

import random


class RandomSplitter:

    def __init__(self, data, percentage):
        """percentage is the percentage of data allocated for training
        and takes values between 0 and 1
        """
        # percentage goes between 0 and 1
        assert(percentage >= 0)
        assert(percentage <= 1)

        if percentage <= 0.5:
            # pick the train data as it is smaller than the test data
            self._train_data = self._subset(data, percentage)
            self._test_data = self._get_residual(data, self._train_data)
        else:
            # pick the test data as it is smaller than the train data
            self._test_data = self._subset(data, 1.0 - percentage)
            self._train_data = self._get_residual(data, self._test_data)

    def get_test_data(self):
        return self._test_data

    def get_train_data(self):
        return self._train_data

    def _subset(self, data, percentage):
        max_n = int(round(len(data) * percentage))

        subset = []
        while len(subset) < max_n:
            choice = random.choice(data)
            if choice not in subset:
                subset.append(choice)

        return subset

    def _get_residual(self, data, test_data):
        return [x for x in data if x not in test_data]
