import argparse

# You can modify or add items as needed.
def parse():
    parser = argparse.ArgumentParser(description='FedAvg')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--k_fold', type=int, default=5, help='the fold number')

    parser.add_argument('--minibatch_size', type=int, default=4, help='batch size')

    parser.add_argument('--num_epochs', type=int, default=5, help='local epochs')

    parser.add_argument('--num_iters', type=int, default=10, help='communication rounds')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--data_dir', type=str, default='./demo_data/data', help='root directory of data')

    parser.add_argument('--label_dir', type=str, default='./demo_data/label', help='root directory of labels')

    parser.add_argument('--save_dir', type=str, default='./results', help='root directory of results')

    argv = parser.parse_args()
    return argv