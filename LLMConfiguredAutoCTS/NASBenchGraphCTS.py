# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import torch
from .NASBenchGraphSource.hpo import HP, Arch
from .NASBenchGraphSource.worker import Worker
from .st_net import Network
import numpy as np
from .dataloader import *
from .evaluation import *
import csv
import os
import time
import h5py
import sys
from types import SimpleNamespace
os.chdir(sys.path[0])
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(model, args, data_dir, dag, ops,cuda_number):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    dataloader = generate_data(data_dir, args.task, args.seq_len, args.output_len, args.in_dim, args.datatype, args.batch_size, args.ratio, test_batch_size=1)
    scaler = dataloader['scaler']
    train_metrics_list = []
    valid_metrics_list = []
    test_metrics_list = []
    for epoch_num in range(args.epochs):
        print(f'epoch num: {epoch_num}')
        model = model.train()

        dataloader['train_loader'].shuffle()
        t2 = time.time()
        train_loss = []
        train_rmse = []
        train_mape = []
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x = torch.Tensor(x).to(cuda_number)

            x = x.transpose(1, 3)

            y = torch.Tensor(y).to(cuda_number)  # [64, 12, 207, 2]

            y = y.transpose(1, 3)[:, 0, :, :]
            optimizer.zero_grad()
            output = model(x)  # [64, 12, 207, 1]
            output = output.transpose(1, 3)
            y = torch.unsqueeze(y, dim=1)
            predict = scaler.inverse_transform(output)  # unnormed x

            loss = masked_mae(predict, y, 0.0)  # y也是unnormed
            train_loss.append(loss.item())
            rmse = masked_rmse(predict, y, 0.0)
            train_rmse.append(rmse.item())
            mape = masked_mape(predict, y, 0.0)
            train_mape.append(mape.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        train_metrics_list.append((np.mean(train_loss), np.mean(train_rmse), np.mean(train_mape)))
        print(f'train epoch time: {time.time() - t2}')

        # eval
        with torch.no_grad():
            model = model.eval()

            valid_loss = []
            valid_rmse = []
            valid_mape = []
            for i, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                x = torch.Tensor(x).to(cuda_number)
                x = x.transpose(1, 3)
                y = torch.Tensor(y).to(cuda_number)
                y = y.transpose(1, 3)[:, 0, :, :]  # [64, 207, 12]

                output = model(x)
                output = output.transpose(1, 3)  # [64, 1, 207, 12]
                y = torch.unsqueeze(y, dim=1)
                predict = scaler.inverse_transform(output)

                loss = masked_mae(predict, y, 0.0)
                rmse = masked_rmse(predict, y, 0.0)
                mape = masked_mape(predict, y, 0.0)
                valid_loss.append(loss.item())
                valid_rmse.append(rmse.item())
                valid_mape.append(mape.item())
            valid_metrics_list.append((np.mean(valid_loss), np.mean(valid_rmse), np.mean(valid_mape)))
            print((np.mean(valid_loss), np.mean(valid_rmse), np.mean(valid_mape)))

        with torch.no_grad():
            model = model.eval()

            y_p = []
            y_t = torch.Tensor(dataloader['y_test']).to(cuda_number)
            y_t = y_t.transpose(1, 3)[:, 0, :, :]
            for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                x = torch.Tensor(x).to(cuda_number)
                x = x.transpose(1, 3)

                output = model(x)
                output = output.transpose(1, 3)  # [64, 1, 207, 12]
                y_p.append(output.squeeze(1))

            y_p = torch.cat(y_p, dim=0)
            y_p = y_p[:y_t.size(0), ...]

            amae = []
            amape = []
            armse = []
            rrse = []
            corr = []
            if args.task == 'multi':
                for i in range(args.seq_len):
                    pred = scaler.inverse_transform(y_p[:, :, i])
                    real = y_t[:, :, i]

                    metrics = metric(pred, real)
                    print(f'{i + 1}, MAE:{metrics[0]}, MAPE:{metrics[1]}, RMSE:{metrics[2]}')
                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])
            else:
                pred = scaler.inverse_transform(y_p)
                real = y_t
                metrics = single_step_metric(pred, real)
                print(f'{i + 1}, RRSE:{metrics[0]}, CORR:{metrics[1]}')
                rrse.append(metrics[0])
                corr.append(metrics[1])

            test_metrics_list.append((np.mean(amae), np.mean(armse), np.mean(amape), np.mean(rrse), np.mean(corr)))
            if args.task == 'multi':
                print(f'On average over {args.seq_len} horizons, '
                      f'Test MAE: {np.mean(amae)}, Test MAPE: {np.mean(amape)}, Test RMSE: {np.mean(armse)}')
            else:
                print(
                    f'Test RRSE: {np.mean(rrse)}, Test CORR: {np.mean(np.mean(corr))}')
                
    if args.error_metrics == 'mae' :
         best_performance = np.mean(amae) 

    detailed_infos = {
                "ops": ops,
                "link": dag,
                "perf": best_performance
            }

    return detailed_infos
    

def run_gnn_experiment(dataset_name, dag, ops, cuda_number,hp=None):

    args = SimpleNamespace(layers=4, steps=4, num_nodes=170, randomadj=True, in_dim=1, hid_dim=32, output_len=12, task='multi', cuda=True, lr=0.001, weight_decay=1e-4,epochs=10, ratio=[0.7, 0.1, 0.2], datatype='h5' ,seq_len=12, batch_size=64, error_metrics='mae')

    datasets_no_matrix = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "solar"]
    if dataset_name in datasets_no_matrix:
        data_dir = os.path.join(f'/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/{dataset_name}/{dataset_name}.csv')
        print("data_dir:",data_dir)
        df = pd.read_csv(data_dir)  
        if dataset_name in["exchange_rate", "solar"]:
            feature_count = df.shape[1] 
        else :
            feature_count = df.shape[1] - 1
        adj_mx = np.zeros((feature_count, feature_count))
        #adj_mx = np.zeros((args.num_nodes, args.num_nodes))
    elif dataset_name == "METR-LA" :
        data_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/METR-LA/metr-la.h5')
        adj_dir = '/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/METR-LA/adj_mx.pkl'
        _, _, adj_mx = load_adj(adj_dir)
        with h5py.File(data_dir, 'r') as f:
            for group_name in f:
                group = f[group_name]
                for dataset_name in group:
                    dataset = group[dataset_name]
                    feature_count = dataset.shape[0]
                    break
    elif dataset_name == "PEMS-BAY" :
        data_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS-BAY/pems-bay.h5')
        adj_dir = '/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS-BAY/adj_mx.pkl'
        _, _, adj_mx = load_adj(adj_dir)
        with h5py.File(data_dir, 'r') as f:
            for group_name in f:
                group = f[group_name]
                for dataset_name in group:
                    dataset = group[dataset_name]
                    feature_count = dataset.shape[0]
                    break
    elif dataset_name == "PEMS03" :
        data_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS03/PEMS03.npz')
        adj_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS03/PEMS03.csv')
        data = np.load(data_dir)
        array_name = 'data' 
        if array_name in data:
            array = data[array_name]
            feature_count = array.shape[1]
            print("feature_count:",feature_count)
        adj_mx = get_adj_matrix(adj_dir, feature_count, id_filename='/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS03/PEMS03.txt')
        print("type(adj_mx):",type(adj_mx))
    elif dataset_name == "PEMS04" :
        data_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS04/PEMS04.npz')
        adj_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS04/PEMS04.csv')
        data = np.load(data_dir)
        array_name = 'data' 
        if array_name in data:
            array = data[array_name]
            feature_count = array.shape[1]
        adj_mx = get_adj_matrix(adj_dir, feature_count)
    elif dataset_name == "PEMS07" :
        data_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS07/PEMS07.npz')
        adj_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS07/PEMS07.csv')
        data = np.load(data_dir)
        array_name = 'data' 
        if array_name in data:
            array = data[array_name]
            feature_count = array.shape[1]
        adj_mx = get_adj_matrix(adj_dir, feature_count)
    elif dataset_name == "PEMS08" :
        data_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS08/PEMS08.npz')
        adj_dir = os.path.join('/root/DesiGNN-copy2/datasets/Unseen datasets/Spatiotemporal-datasets/PEMS08/PEMS08.csv')
        data = np.load(data_dir)
        array_name = 'data' 
        if array_name in data:
            array = data[array_name]
            feature_count = array.shape[1]
        adj_mx = get_adj_matrix(adj_dir, feature_count)
    model = Network(adj_mx, args, dag,feature_count,cuda_number)
    if args.cuda:
        #model = model.cuda()
        model = model.to(cuda_number)
    print(args)
    detailed_infos= run(model, args, data_dir, dag, ops,cuda_number)


    return detailed_infos

