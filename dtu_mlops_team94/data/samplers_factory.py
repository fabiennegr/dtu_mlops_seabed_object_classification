
import torch

class SamplersFactory:

    @staticmethod
    def create(dataset, val_split=0.2, test_split=0.2):
        """Split the dataset into train, validation and test sets.

        Args:
            dataset: dataset object
            val_split: fraction of the dataset to use for validation
            test_split: fraction of the dataset to use for testing

        Returns:
            train_sampler: sampler for the training set
            val_sampler: sampler for the validation set
            test_sampler: sampler for the test set
        """
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int((1 - val_split - test_split) * len(dataset))
        )
        val_sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=int(val_split * len(dataset)))
        test_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, num_samples=int(test_split * len(dataset))
        )

        return train_sampler, val_sampler, test_sampler
