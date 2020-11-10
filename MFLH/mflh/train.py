import os
import argparse
import warnings
import utils.data_load.image as dataset
import utils.model.mflh as model

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

param = argparse.ArgumentParser(description='MFLH')
param.add_argument('--gpus', default='0', type=str)
param.add_argument('--dataset', default='cifar10', type=str)
param.add_argument('-b', '--batch-size', default=64, type=int)
param.add_argument('-vb', '--test-batch-size', default=16, type=int)
param.add_argument('--lr', default=1e-5, type=float)
param.add_argument('--epochs', default=100, type=int)
param.add_argument('--output-dim', default=12, type=int)
param.add_argument('--scale-factor', default=8, type=int)
param.add_argument('--triplet-margin', default=30, type=float)
param.add_argument('--strategy', default='all', choices=['hard', 'all'])
param.add_argument('--alpha', default=0.0, type=int)
param.add_argument('--eta', default=0.8, type=int)
param.add_argument('--gamma', default=0.1, type=int)
param.add_argument('--size', default=227, type=int)
param.add_argument('--decay-step', default=20, type=int)
param.add_argument('--decay-rate', default=0.96, type=int)
param.add_argument('--model-weights', type=str, default='../')
param.add_argument('--data-dir', default="../", type=str)
param.add_argument('--output-dir', default='../', type=str)
param.add_argument('--pre-model-path', default='../', type=str)
param.add_argument('--dist', default='euclidean2')

args = param.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
class_dims = {'cifar10': 10, 'nuswide_81': 21}
Nums = {'cifar10': 59000, 'nuswide_81': 5000}
args.number = Nums[args.dataset]
args.label_dim = class_dims[args.dataset]

args.img_train = os.path.join(args.data_dir, args.dataset, "train.txt")
args.img_test = os.path.join(args.data_dir, args.dataset, "test.txt")
args.img_database = os.path.join(args.data_dir, args.dataset, "database.txt")

data_root = os.path.join(args.data_dir, args.dataset)
query_img, database_img = dataset.import_test(data_root, args.img_test, args.img_database, args)

train_img = dataset.import_train(data_root, args.img_train, args)
model_weights = model.train(train_img, database_img, query_img, args)
args.model_weights = model_weights
