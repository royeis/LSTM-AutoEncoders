import numpy as np

def generate_random_sequence():
    arr = np.random.uniform(0.0, 1.0, 50)
    diff = (np.sum(arr) - (0.5 * 50))
    args = np.argsort(arr)
    if diff > 0:
        arr[args[-26:-1]] -= diff / 25
    else:
        arr[args[:25]] -= diff / 25

    return arr


def create_toy_data(n_seqs=10000):
    dataset = []
    train_size = int(0.6*n_seqs)
    validate_size = int(0.8 * n_seqs)
    for i in range(n_seqs):
        dataset.append(generate_random_sequence())
    # return np.expand_dims(dataset, 2)
    dataset = np.expand_dims(dataset, axis=2)
    return dataset[:train_size], dataset[train_size:validate_size], dataset[validate_size:]


if __name__ == '__main__':
    x_train, x_validate, x_test = create_toy_data()
    print('done')