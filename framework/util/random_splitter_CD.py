# This class splits a dataset in a random way using percentages allocated for
# each part. It is used when training Chris Davey's model and needing a
# training dataset and test dataset.

import random
import framework.data.formats as formats


class RandomSplitterCD:

    def __init__(self, plot_data, genotype_data, percentage):
        """percentage is the percentage of data allocated for training
        and takes values between 0 and 1
        """
        # percentage goes between 0 and 1
        assert(percentage >= 0)
        assert(percentage <= 1)

        if percentage <= 0.5:
            # pick the train data as it is smaller than the test data
            self._train_data = self._subset(genotype_data, percentage)
            self._test_data = self._get_residual(genotype_data,
                                                 self._train_data)
        else:
            # pick the test data as it is smaller than the train data
            self._test_data = self._subset(genotype_data, 1.0 - percentage)
            self._train_data = self._get_residual(genotype_data,
                                                  self._test_data)

        # finally get the plot level for the training data
        self._train_data = self._get_plot_level(self._train_data, plot_data)
        import ipdb
        ipdb.set_trace()

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

    def _get_plot_level(self, genotype_data, plot_data):
        output_data = []
        for entry in genotype_data:
            match = [x for x in plot_data if
                     x[formats.GENOTYPE] == entry[formats.GENOTYPE] and
                     x[formats.DATE] == entry[formats.DATE]]
            if len(match) != 3:
                import ipdb
                ipdb.set_trace()

            output_data += match

        return match
