from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import numpy as np


class TrainerConfig:
    test_size = 0.2
    fold_count = 4
    batch_size = 64
    epochs = 2


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def initialize(self, x, y):
        pass


class TestTrainSplitTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.train_x, self.test_x, self.train_y, self.test_y = [None] * 4

    def initialize(self, x, y):
        x = self.model.data_processor.preprocess(x)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            x.values, y, test_size=self.config.test_size, random_state=0
        )
        self.train_x = self.model.data_processor.tokenize(self.train_x)
        self.test_x = self.model.data_processor.tokenize(self.test_x)

    def train(self):
        self.train_x = self.model._prepare(self.train_x)
        self.test_x = self.model._prepare(self.test_x)
        training_history = self.model.train(train_x=self.train_x, train_y=self.train_y, test_x=self.test_x,
                                            test_y=self.test_y, config=self.config)
        return training_history


class KFoldTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.fold = None
        self.x = None
        self.y = None

    def initialize(self, x, y):
        x = self.model.data_processor.preprocess(x)
        self.x = self.model.data_processor.tokenize(x)
        self.y = y

    def train(self):
        training_histories = []
        self.fold = KFold(n_splits=self.config.fold_count, shuffle=True, random_state=108)
        for i, (train_idx, test_idx) in enumerate(self.fold.split(np.arange(self.y.shape[0]))):
            train_x = self.model._prepare(self.x, train_idx)
            train_y = self.y[train_idx]

            test_x = self.model._prepare(self.x, test_idx)
            test_y = self.y[test_idx]
            training_history = self.model.train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                                                config=self.config)
            training_histories.append(training_history)
        return training_histories
