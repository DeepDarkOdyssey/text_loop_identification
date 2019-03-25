from random import randint, random, seed
from sklearn.model_selection import train_test_split
import numpy as np

NUM_RANGE = 10

seed(20190325)
np.random.seed(20190325)


def generate_pos_sample():
    prefix_length = randint(0, 20)
    postfix_length = randint(0, 20)
    prefix = np.random.randint(NUM_RANGE, size=(prefix_length))
    postfix = np.random.randint(NUM_RANGE, size=(postfix_length))

    loop_length = randint(15, 40)
    loop_times = randint(5, 20)
    loop = np.random.randint(NUM_RANGE, size=(loop_length))

    loop_nums = None
    for _ in range(loop_times):
        rand_length = randint(0, 20)
        rand_nums = np.random.randint(NUM_RANGE, size=(rand_length))
        if loop_nums is None:
            loop_nums = np.concatenate([loop, rand_nums])
        else:
            loop_nums = np.concatenate([loop_nums, loop, rand_nums])
    
    return np.concatenate([prefix, loop_nums, postfix])


def generate_neg_sample():
    min_len = 75
    max_len = 1240
    return np.random.randint(NUM_RANGE, size=randint(min_len, max_len))


def build_dataset(dataset_size, neg_frac=0.5):

    dataset = []
    for _ in range(dataset_size):
        if random() > neg_frac:
            dataset.append((generate_pos_sample(), 1))
        else:
            dataset.append((generate_neg_sample(), 0))
    
    return dataset
        

def write_data_to_file(dataset, file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        for num_seq, label in dataset:
            f.write(f'{" ".join([str(n) for n in num_seq])}\t{label}\n')


def build_datasets(train_size, dev_size, test_size, neg_frac=0.5):
    print('Building datasets')
    train_dev_set = build_dataset(train_size + dev_size, neg_frac)
    train_set, dev_set = train_test_split(train_dev_set, test_size=dev_size)
    test_set = build_dataset(test_size, neg_frac=neg_frac)
    print('Writing dataset to files')
    write_data_to_file(train_set, 'data/train.tsv')
    write_data_to_file(dev_set, 'data/dev.tsv')
    write_data_to_file(test_set, 'data/test.tsv')
    print('Finished')
    

if __name__ == "__main__":
    build_datasets(10000, 1000, 1000)
