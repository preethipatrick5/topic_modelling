import os
import abc
import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel, TFDistilBertForSequenceClassification, \
    TFBertForSequenceClassification, BertTokenizer
import numpy as np


class ModelConfig:
    max_length = 200
    bert_name = "bert-base-uncased"
    num_labels = 6
    optimizer = tf.keras.optimizers.Adam(0.00001)
    loss_function = "binary_crossentropy"
    metrics = ["accuracy"]


class DataProcessor:

    def __init__(self, tokenizer, processors, max_length):
        self.tokenizer = tokenizer
        self.processors = processors
        self.max_length = max_length

    def preprocess(self, data):
        for processor in self.processors:
            data = data.apply(processor)
        return data

    def tokenize(self, data):
        if type(data) is not list:
            data = list(data)
        return self.tokenizer(data, add_special_tokens=True,
                              max_length=self.max_length, return_tensors="tf",
                              return_token_type_ids=True, return_attention_mask=True,
                              truncation=True, padding="max_length")

    def tokenize_plus(self, data):
        if type(data) is not list:
            data = list(data)
        return self.tokenizer.encode_plus(data, add_special_tokens=True,
                                          max_length=self.max_length, return_tensors="tf",
                                          return_token_type_ids=True, return_attention_mask=True,
                                          truncation=True, padding="max_length")

    def add_processor(self, processor):
        self.processors.append(processor)


class Model(abc.ABC):

    def __init__(self, *, config, version="1"):
        self.model = None
        self.config = config
        self.version = version
        self.data_processor = None
        self.tokenizer = None

    @abc.abstractmethod
    def build(self, processors):
        pass

    def build_data_processor(self, processors):
        self.data_processor = DataProcessor(self.tokenizer, processors=processors, max_length=self.config.max_length)
        return self.data_processor

    @abc.abstractmethod
    def predict(self, x):
        pass

    @staticmethod
    def _prepare(data, ids=None, from_bert_encoded=True):
        if from_bert_encoded:
            data["input_ids"] = np.array(data["input_ids"])
            data["token_type_ids"] = np.array(data["token_type_ids"])
            data["attention_mask"] = np.array(data["attention_mask"])
        if ids is not None:
            return [
                data['input_ids'][ids],
                data['token_type_ids'][ids],
                data['attention_mask'][ids]
            ]
        else:
            return [
                data['input_ids'],
                data['token_type_ids'],
                data['attention_mask']
            ]

    def train(self, train_x, train_y, test_x, test_y, config):
        training_history = self.model.fit(
            x=train_x, y=train_y,
            epochs=config.epochs, batch_size=config.batch_size,
            validation_data=(test_x, test_y)
        )
        return training_history

    def name(self):
        return "{}_{}".format(self.__class__.__name__, self.version)

    def save(self, directory, version=None):
        model_name = "{}_{}".format(self.name(), version) if version else self.name()
        self.model.save_weights(os.path.join(directory, model_name), save_format="h5")

    def load(self, directory, version=None):
        model_name = "{}_{}".format(self.name(), version) if version else self.name()
        self.model.load_weights(os.path.join(directory, model_name))


class SimpleBertForSequenceClassification(Model):

    def __init__(self, *, config):
        super().__init__(config=config)

    def build(self, processors=None):
        if processors is None:
            processors = []
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)
        self.data_processor = self.build_data_processor(processors=processors)
        self.model = TFBertForSequenceClassification.from_pretrained(self.config.bert_name,
                                                                     num_labels=self.config.num_labels)
        self.model.compile(optimizer=self.config.optimizer, loss=self.config.loss_function, metrics=self.config.metrics)
        return self

    @staticmethod
    def _prepare(data, ids=None, from_bert_encoded=True):
        if from_bert_encoded:
            data["input_ids"] = np.array(data["input_ids"])
            data["attention_mask"] = np.array(data["attention_mask"])
        if ids is not None:
            return [
                data['input_ids'][ids],
                data['attention_mask'][ids]
            ]
        else:
            return [
                data['input_ids'],
                data['attention_mask']
            ]

    def predict(self, x):
        x = self.data_processor.preprocess(x)
        tokenized = self.data_processor.tokenize(x)
        results = self.model(tokenized)
        return tf.nn.sigmoid(results.logits).numpy()


class ModifiedBertForSequenceClassification(Model):

    def __init__(self, *, config):
        super().__init__(config=config)

    def build(self, processors=None):
        if processors is None:
            processors = []
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)
        self.data_processor = self.build_data_processor(processors=processors)
        input_ids = tf.keras.layers.Input(shape=(self.config.max_length,), name='input_ids', dtype='int32')
        attention_mask = tf.keras.layers.Input(shape=(self.config.max_length,), name='attention_mask', dtype='int32')
        transformer_model = TFBertForSequenceClassification.from_pretrained(self.config.bert_name,
                                                                            output_hidden_states=True)

        transformer = transformer_model([input_ids, attention_mask])
        hidden_states = transformer[1]

        hidden_states_size = 4
        hidden_states_ind = list(range(-hidden_states_size, 0, 1))

        selected_hidden_states = tf.keras.layers.concatenate(tuple([hidden_states[i] for i in hidden_states_ind]))

        output = tf.keras.layers.Dense(128, activation='relu')(selected_hidden_states)
        output = tf.keras.layers.Flatten()(output)
        output = tf.keras.layers.Dense(self.config.num_labels, activation='sigmoid')(output)
        self.model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
        self.model.compile(optimizer=self.config.optimizer, loss=self.config.loss_function, metrics=self.config.metrics)
        return self

    @staticmethod
    def _prepare(data, ids=None, from_bert_encoded=True):
        if from_bert_encoded:
            data["input_ids"] = np.array(data["input_ids"])
            data["attention_mask"] = np.array(data["attention_mask"])
        if ids is not None:
            return [
                data['input_ids'][ids],
                data['attention_mask'][ids]
            ]
        else:
            return [
                data['input_ids'],
                data['attention_mask']
            ]

    def predict(self, x):
        x = self.data_processor.preprocess(x)
        tokenized = self.data_processor.tokenize(x)
        results = self.model.predict([tokenized["input_ids"], tokenized["attention_mask"]])
        return results


class BertForSequenceClassificationWithDNN(Model):

    def __init__(self, *, config):
        super().__init__(config=config)

    def build(self, processors=None):
        if processors is None:
            processors = []
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)
        self.data_processor = self.build_data_processor(processors=processors)
        bert = TFBertForSequenceClassification.from_pretrained(self.config.bert_name)

        input_ids = tf.keras.layers.Input(shape=(self.config.max_length,), name='input_ids', dtype='int32')
        input_type = tf.keras.layers.Input(shape=(self.config.max_length,), name='token_type_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(self.config.max_length,), name='attention_mask', dtype='int32')

        embeddings = bert([input_ids, mask, input_type])

        x = tf.keras.layers.Dense(32, activation='relu')(embeddings[0])
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(16, activation='sigmoid')(x)
        x = tf.keras.layers.Dropout(0.01)(x)
        y = tf.keras.layers.Dense(self.config.num_labels, activation='sigmoid', name='outputs')(x)

        self.model = tf.keras.Model(inputs=[input_ids, input_type, mask], outputs=y)

        self.model.compile(optimizer=self.config.optimizer, loss=self.config.loss_function, metrics=self.config.metrics)
        return self

    def predict(self, x):
        x = self.data_processor.preprocess(x)
        tokenized = self.data_processor.tokenize(x)
        results = self.model.predict([tokenized["input_ids"], tokenized["token_type_ids"], tokenized["attention_mask"]])
        return results


class BertForSequenceClassificationWithLSTM(Model):

    def __init__(self, *, config):
        super().__init__(config=config)

    def build(self, processors=None):
        if processors is None:
            processors = []
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)
        self.data_processor = self.build_data_processor(processors=processors)
        bert = TFBertForSequenceClassification.from_pretrained(self.config.bert_name)

        input_ids = tf.keras.layers.Input(shape=(self.config.max_length,), name='input_ids', dtype='int32')
        input_type = tf.keras.layers.Input(shape=(self.config.max_length,), name='token_type_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(self.config.max_length,), name='attention_mask', dtype='int32')

        embeddings = bert([input_ids, mask, input_type])

        x = tf.keras.layers.LSTM(128)(embeddings)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.01)(x)
        y = tf.keras.layers.Dense(self.config.num_labels, activation='sigmoid', name='outputs')(x)

        self.model = tf.keras.Model(inputs=[input_ids, input_type, mask], outputs=y)

        self.model.compile(optimizer=self.config.optimizer, loss=self.config.loss_function, metrics=self.config.metrics)
        return self

    def predict(self, x):
        x = self.data_processor.preprocess(x)
        tokenized = self.data_processor.tokenize_plus(x)
        results = self.model.predict([tokenized["input_ids"], tokenized["token_type_ids"], tokenized["attention_mask"]])
        return results


class BertToSingleLayerNeuralNetwork(Model):

    def __init__(self, *, config):
        super().__init__(config=config)

    def build(self, processors=None):
        if processors is None:
            processors = []
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)
        self.data_processor = self.build_data_processor(processors=processors)
        bert = TFBertModel.from_pretrained(self.config.bert_name)

        input_ids = tf.keras.layers.Input(shape=(self.config.max_length,), name='input_ids', dtype='int32')
        input_type = tf.keras.layers.Input(shape=(self.config.max_length,), name='token_type_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(self.config.max_length,), name='attention_mask', dtype='int32')

        embeddings = bert([input_ids, mask, input_type])[0]

        y = tf.keras.layers.Dense(self.config.num_labels, activation='sigmoid', name='outputs')(embeddings[:, 0, :])

        self.model = tf.keras.Model(inputs=[input_ids, input_type, mask], outputs=y)

        self.model.compile(optimizer=self.config.optimizer, loss=self.config.loss_function, metrics=self.config.metrics)
        return self

    def predict(self, x):
        x = self.data_processor.preprocess(x)
        tokenized = self.data_processor.tokenize(x)
        return self.model.predict([tokenized["input_ids"], tokenized["token_type_ids"], tokenized["attention_mask"]])


class BertWithLSTM(Model):

    def __init__(self, *, config):
        super().__init__(config=config)

    def build(self, processors=None):
        if processors is None:
            processors = []
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)
        self.data_processor = self.build_data_processor(processors=processors)
        bert = TFBertModel.from_pretrained(self.config.bert_name)

        input_ids = tf.keras.layers.Input(shape=(self.config.max_length,), name='input_ids', dtype='int32')
        input_type = tf.keras.layers.Input(shape=(self.config.max_length,), name='token_type_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(self.config.max_length,), name='attention_mask', dtype='int32')

        embeddings = bert([input_ids, mask, input_type])[0]

        x = tf.keras.layers.LSTM(128)(embeddings)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.01)(x)
        y = tf.keras.layers.Dense(self.config.num_labels, activation='sigmoid', name='outputs')(x)

        self.model = tf.keras.Model(inputs=[input_ids, input_type, mask], outputs=y)

        self.model.compile(optimizer=self.config.optimizer, loss=self.config.loss_function, metrics=self.config.metrics)
        return self

    def predict(self, x):
        x = self.data_processor.preprocess(x)
        tokenized = self.data_processor.tokenize(x)
        return self.model.predict([tokenized["input_ids"], tokenized["token_type_ids"], tokenized["attention_mask"]])
