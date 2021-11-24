import unittest
from ds.load_data import train_test_split_k_shot, get_transforms
from ds.dataset import create_data_loader


class MyTestCase(unittest.TestCase):
    def test_load_data(self):
        root = '../../data'
        img_dir = 'classification_20_clean'
        k = 1
        num_of_exp = 1
        batch_size = 32
        train, test = train_test_split_k_shot(root, img_dir, k, num_of_exp)
        transforms = get_transforms()
        print(test.shape)
        test_loader = create_data_loader(annotations_file=test, root=root, transform=transforms['test'],
                                         batch_size=batch_size)
        train_loader = create_data_loader(annotations_file=train, root=root, transform=transforms['train'],
                                          batch_size=batch_size)
        print(train.shape, test.shape)
        self.assertEqual(True, True)  # add assertion here

