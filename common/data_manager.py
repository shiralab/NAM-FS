import os
import sys
import gzip
import numpy as np

from pathlib import Path
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import openml


class DataManager(object):
    def __init__(self, dataset_name, scalar='Standard', test_size=0.2, seed=0, downloads_path='./dataset/downloads/',
                 npz_path='./dataset/npz/'):
        self.name = dataset_name
        self.seed = seed
        self.npz_file_name = str(Path(npz_path) / self.name)

        # Load npz file
        if os.path.isfile(self.npz_file_name + '.npz'):
            print('Loading npz file from ' + self.npz_file_name + '.npz' + ' ...')
            npz_data = np.load(self.npz_file_name + '.npz')
            self.x_train = npz_data['x_train']
            self.y_train = npz_data['y_train']
            self.x_test = npz_data['x_test']
            self.y_test = npz_data['y_test']
            self.feature_names = npz_data['feature_names']
            self.target_names = npz_data['target_names']
            self.n_class, self.n_features = len(self.target_names), self.x_train.shape[1]

        # Load dataset
        else:
            print('Loading dataset (' + self.name + ')...')
            if self.name == 'breast':
                # From scikit-learn dataset
                dataset = load_breast_cancer()
                x_data = dataset.data.astype(np.float32)
                y_data = dataset.target.astype(np.int64)
                self.feature_names = dataset.feature_names
                self.target_names = dataset.target_names
                self.n_class = len(self.target_names)  # 2
                self.n_features = x_data.shape[1]  # 30
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                        test_size=test_size,
                                                                                        random_state=self.seed,
                                                                                        stratify=y_data)

            elif self.name == 'covertype':
                # From scikit-learn dataset
                dataset = fetch_covtype()
                x_data = dataset.data.astype(np.float32)
                y_data = dataset.target.astype(np.int64) - 1
                self.feature_names = dataset.feature_names
                self.target_names = np.array(['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow',
                                              'Aspen', 'Douglas-fir', 'Krummholz'])
                self.n_class = len(self.target_names)  # 7
                self.n_features = x_data.shape[1]  # 54
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                        test_size=test_size,
                                                                                        random_state=self.seed,
                                                                                        stratify=y_data)

            elif self.name == 'spambase':
                # From OpenML
                dataset = openml.datasets.get_dataset(44)  # Get dataset by ID
                data, _, _, names = dataset.get_data(dataset_format='dataframe')
                x_data = data.iloc[:, :-1].to_numpy(dtype=np.float32)
                y_data = data['class'].to_numpy(dtype=np.int64)
                self.feature_names = np.array(names[:-1])
                self.target_names = np.array(['not_spam', 'spam'])
                self.n_class = len(self.target_names)  # 2
                self.n_features = x_data.shape[1]  # 57
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                        test_size=test_size,
                                                                                        random_state=self.seed,
                                                                                        stratify=y_data)

            elif self.name == 'har':
                # From OpenML
                dataset = openml.datasets.get_dataset(1478)  # Get dataset by ID
                data, _, _, names = dataset.get_data(dataset_format='dataframe')
                x_data = data.iloc[:, :-1].to_numpy(dtype=np.float32)
                y_data = data['Class'].to_numpy(dtype=np.int64) - 1  # to start zero
                self.feature_names = np.array(names[:-1])
                self.target_names = np.array(['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING',
                                              'STANDING', 'LAYING'])
                self.n_class = len(self.target_names)  # 6
                self.n_features = x_data.shape[1]  # 561
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                        test_size=test_size,
                                                                                        random_state=self.seed,
                                                                                        stratify=y_data)

            elif self.name == 'isolet':
                # From OpenML
                dataset = openml.datasets.get_dataset(300)  # Get dataset by ID
                data, _, _, names = dataset.get_data(dataset_format='dataframe')
                x_data = data.iloc[:, :-1].to_numpy(dtype=np.float32)
                y_data = data['class'].to_numpy(dtype=np.int64) - 1  # to start zero
                self.feature_names = np.array(names[:-1])
                self.target_names = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                                              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
                self.n_class = len(self.target_names)  # 26
                self.n_features = x_data.shape[1]  # 617
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                        test_size=test_size,
                                                                                        random_state=self.seed,
                                                                                        stratify=y_data)

            elif self.name == 'epsilon':
                # From csv file
                # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
                x_train, y_train, x_test, y_test = load_svmlight_files((Path(downloads_path) / 'epsilon_normalized',
                                                                        Path(downloads_path) / 'epsilon_normalized.t'))

                self.x_train = x_train.toarray().astype(np.float32)
                self.x_test = x_test.toarray().astype(np.float32)
                self.y_train = (y_train / 2. + 0.5).astype(np.int64)
                self.y_test = (y_test / 2. + 0.5).astype(np.int64)
                self.feature_names = np.array([f'{i+1}' for i in range(self.x_train.shape[1])])
                self.target_names = np.array(['0', '1'])
                self.n_class = len(self.target_names)  # 2
                self.n_features = self.x_train.shape[1]  # 2000

            elif self.name == 'guillermo':
                # From OpenML
                dataset = openml.datasets.get_dataset(41159)  # Get dataset by ID
                data, _, _, names = dataset.get_data(dataset_format='dataframe')
                x_data = data.iloc[:, 1:].to_numpy(dtype=np.float32)
                y_data = data['class'].to_numpy(dtype=np.int64)
                self.feature_names = np.array(names[1:])
                self.target_names = np.array(['0', '1'])
                self.n_class = len(self.target_names)  # 2
                self.n_features = x_data.shape[1]  # 4296
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                        test_size=test_size,
                                                                                        random_state=self.seed,
                                                                                        stratify=y_data)

            elif self.name == 'gisette':
                # From OpenML
                dataset = openml.datasets.get_dataset(41026)  # Get dataset by ID
                data, _, _, names = dataset.get_data(dataset_format='dataframe')
                x_data = data.iloc[:, :-1].to_numpy(dtype=np.float32)
                y_data = data['class'].to_numpy(dtype=np.int64)
                self.feature_names = np.array(names[:-1])
                self.target_names = np.array(['0', '1'])
                self.n_class = len(self.target_names)  # 2
                self.n_features = x_data.shape[1]  # 5000
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                        test_size=test_size,
                                                                                        random_state=self.seed,
                                                                                        stratify=y_data)

            elif self.name == 'fashion-mnist':
                # From files
                labels_path = os.path.join(downloads_path, 'train-labels-idx1-ubyte.gz')
                images_path = os.path.join(downloads_path, 'train-images-idx3-ubyte.gz')
                with gzip.open(labels_path, 'rb') as lbpath:
                    self.y_train = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8).astype(np.int64)
                with gzip.open(images_path, 'rb') as imgpath:
                    self.x_train = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(self.y_train), 784).astype(np.float32)

                labels_path = os.path.join(downloads_path, 't10k-labels-idx1-ubyte.gz')
                images_path = os.path.join(downloads_path, 't10k-images-idx3-ubyte.gz')
                with gzip.open(labels_path, 'rb') as lbpath:
                    self.y_test = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8).astype(np.int64)
                with gzip.open(images_path, 'rb') as imgpath:
                    self.x_test = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(self.y_test), 784).astype(np.float32)

                self.feature_names = np.array([f'pixel{i+1}' for i in range(784)])
                self.target_names = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                                              'Sneaker', 'Bag', 'Ankle boot'])
                self.n_class = len(self.target_names)  # 10
                self.n_features = self.x_train.shape[1]  # 784

            else:
                print('Invalid Dataset Name')
                sys.exit(1)

            # Save dataset as NumPy binary
            os.makedirs(npz_path, exist_ok=True)
            np.savez(self.npz_file_name, x_train=self.x_train, x_test=self.x_test, y_train=self.y_train,
                     y_test=self.y_test, feature_names=self.feature_names, target_names=self.target_names)

        # Scalar
        self.scalar = StandardScaler() if scalar == 'Standard' else MinMaxScaler((-1, 1))
        self.x_train_scaled = self.scalar.fit_transform(self.x_train)
        self.x_test_scaled = self.scalar.transform(self.x_test)
