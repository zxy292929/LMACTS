from pathlib import Path
import pickle
import numpy as np
import csv
import pandas as pd

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pkl_filename):
    try:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pkl_filename, ':', e)
        raise

    return pkl_data

def get_adj_matrix(distance_df_filename, num_of_vertices, id_filename = None, type_='connectivity'):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':  # 啥意思啊，表里有的就置1？
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be connectivity or distance!")
    return A
    
def generate_data(graph_signal_matrix_name, task, train_len, pred_len, in_dim, type, batch_size, ratio=[0.6, 0.2, 0.2], test_batch_size=None,
                  transformer=None):
    """shape=[num_of_samples, 12, num_of_vertices, 1]"""
    data = data_preprocess(graph_signal_matrix_name, task, train_len, pred_len, in_dim, type, ratio)

    scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar  # 有没有问题？只用一半训练数据的时候

    return data

def data_preprocess(data_path, task, train_len, pred_len, in_dim, type, ratio=[0.6, 0.2, 0.2], transformer=None):
    if type == 'csv':
        origin_data = pd.read_csv(data_path)
        if 'date' in origin_data.columns:
            origin_data.set_index('date', inplace=True)
        origin_data = origin_data.values
        # origin_data = np.expand_dims(origin_data, -1)
        origin_data = dim_uniform(origin_data)

        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)

    elif type == 'txt':
        origin_data = np.loadtxt(data_path, delimiter=',')

        origin_data = np.array(origin_data)
        origin_data = dim_uniform(origin_data)
        # origin_data = np.expand_dims(origin_data, -1)
        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)
    
    elif type == 'npz' or type == 'subset':
        origin_data = np.load(data_path, allow_pickle=True)
        try:  # shape=[17856, 170, 3]
            keys = origin_data.keys()
            if 'train' in keys and 'val' in keys and 'test' in keys:
                data = generate_from_train_val_test(dim_uniform(origin_data['data']), train_len, pred_len, in_dim,
                                                    ratio, transformer)

            elif 'data' in keys:
                length = origin_data['data'].shape[0]
                data = generate_from_data(dim_uniform(origin_data['data']), length, task, train_len, pred_len, in_dim,
                                          ratio,
                                          transformer)

        except:
            length = origin_data.shape[0]
            data = generate_from_data(dim_uniform(origin_data), length, task, train_len, pred_len, in_dim, ratio,
                                      transformer)
    elif type == 'h5':
        origin_data = pd.read_hdf(data_path)
        origin_data = np.array(origin_data)
        origin_data = dim_uniform(origin_data)

        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)
    elif type == 'npy':
        arr = np.load(data_path, allow_pickle=True).astype('float')
        # 获取数组的形状
        nan_mask = np.isnan(arr)

        # 计算沿轴0的均值，但在计算之前检查轴上是否有NaN值
        mean_values = np.where(np.all(nan_mask, axis=2, keepdims=True), 0, np.nanmean(arr, axis=2, keepdims=True))

        # 使用 np.where 将NaN值替换为均值
        arr_filled = np.where(nan_mask, mean_values, arr)

        origin_data = arr_filled
        length = len(origin_data)
        data = generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio)
    return data

def dim_uniform(origin_data):
    if origin_data.ndim == 1:
        data = origin_data.reshape((origin_data.shape[0], 1, 1))
    elif origin_data.ndim == 2:
        data = origin_data.reshape((origin_data.shape[0], origin_data.shape[1], 1))
    else:
        data = origin_data

    return data

def generate_from_data(origin_data, length, task, train_len, pred_len, in_dim, ratio, transformer=None):
    """origin_data shape: [17856, 170, 3]"""
    data = generate_sample(origin_data, task, train_len, pred_len, in_dim)
    train_ratio, val_ratio, test_ratio = ratio
    train_line, val_line = int(length * train_ratio), int(length * (train_ratio + val_ratio))
    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):
        x, y = generate_seq(origin_data[line1: line2], task, train_len, pred_len, in_dim)
        print(x.shape)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')
        # if transformer:  # 啥意思？
        #     x = transformer(x)
        #     y = transformer(y)

    return data

def generate_from_train_val_test(origin_data, train_len, pred_len, in_dim, transformer=None):
    data = {}
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(origin_data[key], train_len, pred_len, in_dim)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')
        # if transformer:  # 啥意思？
        #     x = transformer(x)
        #     y = transformer(y)

    return data

def generate_sample(origin_data, task, train_len, pred_len, in_dim):
    data = {}
    data['origin'] = origin_data
    x, y = generate_seq(origin_data, task, train_len, pred_len, in_dim)
    data['x'] = x.astype('float32')
    data['y'] = y.astype('float32')
    return data

def generate_seq(data, task, train_length, pred_length, in_dim):
    if task == 'multi':
        seq = np.concatenate([np.expand_dims(
            data[i: i + train_length + pred_length], 0)
            for i in range(data.shape[0] - train_length - pred_length + 1)],
            axis=0)[:, :, :, 0: in_dim]
        if train_length == pred_length:
            return np.split(seq, 2, axis=1)
        else:
            return np.split(seq, [train_length], axis=1)
    elif task == 'single':
        return generate_seq_for_single_step(data, train_length, pred_length, in_dim)
    else:
        raise ValueError
    
def generate_seq_for_single_step(data, train_length, pred_index, in_dim):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_index], 0)
        for i in range(data.shape[0] - train_length - pred_index + 1)],
        axis=0)[:, :, :, 0: in_dim]

    X, Y = np.split(seq, [train_length], axis=1)
    Y = Y[:, pred_index - 1:pred_index, :, :]
    return X, Y

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean
    
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        :param shuffle:
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]  # ...代替多个:
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()