import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ps = PorterStemmer()
lm = WordNetLemmatizer()
nltk.download('all')


def replace_latex_math_with(abstract):
    abstract = abstract.strip()
    is_equation = False
    results = abstract.split("$")
    for index, result in enumerate(results):
        if is_equation:
            results[index] = "equation"
        is_equation = not is_equation
    results = " ".join(results)
    starting_index = results.find("\\begin{equation}")
    if starting_index != -1:
        ending_index = results.find("\\end{equation}") + len("\\end{equation}")
        results = results[:starting_index] + " equation " + results[ending_index + 1:]
    return results


def to_corpus(text):
    return text.split()


def stem_sentence(sentence):
    stemmed_sentence = []
    for word in sentence:
        stemmed_sentence.append(ps.stem(word))
        stemmed_sentence.append(" ")
    return " ".join(stemmed_sentence)


def lemmatize_sentence(sentence):
    lemmatized_sentence = []
    for w in sentence:
        lemmatized_sentence.append(lm.lemmatize(w))
        lemmatized_sentence.append(" ")
    return " ".join(lemmatized_sentence)


def clean_stop_words(sentence):
    filtered_sentence = []
    stop_words = set(stopwords.words('english'))
    for w in sentence:
        if not w in stop_words:
            filtered_sentence.append(w)
    return " ".join(filtered_sentence)


# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
def plot_roc(y_test, y_score, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def _draw_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


# https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
def draw_confusion_matrix(cm, class_list):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    for axes, cfs_matrix, label in zip(ax.flatten(), cm, class_list):
        _draw_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.show()


def draw_f1_scores(y_trues, y_preds, list_classes, model_names, figure_size=(10, 7)):
    f1_scores = []
    for y_true, y_pred in zip(y_trues, y_preds):
        f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred, average=None)
        f1_scores.append(f1_score)
    draw_for_metric(f1_scores, list_classes, model_names, figure_size)


def draw_hamming_loss(y_trues, y_preds, list_classes, model_names, figure_size=(10, 7)):
    hamming_losses = []
    for y_true, y_pred in zip(y_trues, y_preds):
        f1_score = metrics.hamming_loss(y_true=y_true, y_pred=y_pred)
        hamming_losses.append(f1_score)
    draw_for_metric(hamming_losses, list_classes, model_names, figure_size)


def draw_for_metric(metrics, list_classes, model_names, figure_size=(15, 7)):
    plt.figure(figsize=figure_size)
    plt.stackplot(list_classes, *metrics, labels=model_names)
    plt.legend(loc='upper left')


def plot_training_result(history, title, figure_size=(8, 4)):
    if "fold_1" in history.keys():
        for h in history.keys():
            plt.plot(history[h]['accuracy'])
            plt.plot(history[h]['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(["{} train".format(h), "{} val".format(h)], loc='upper left')
        plt.show()
        for h in history.keys():
            plt.plot(history[h]['loss'])
            plt.plot(history[h]['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(["{} train".format(h), "{} val".format(h)], loc='upper left')
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
        fig.suptitle(title)
        ax1.plot(history['accuracy'])
        ax1.plot(history['val_accuracy'])
        ax1.set_title('model accuracy')
        ax1.set(xlabel='epoch', ylabel='accuracy')
        ax1.legend(['train', 'val'], loc='upper left')

        ax2.plot(history['loss'])
        ax2.plot(history['val_loss'])
        ax2.set_title('model loss')
        ax1.set(xlabel='epoch', ylabel='loss')
        ax2.legend(['train', 'val'], loc='upper left')
        plt.show()


def training_history_to_dict(training_history):
    if type(training_history) is list:
        history_dict = {}
        for index, history in enumerate(training_history):
            history_dict["fold_{}".format(index + 1)] = history.history
        return history_dict
    else:
        return training_history.history


def evaluate(y_pred, Y, classes, threshold=0.5):
    converted_y_pred = (y_pred > threshold).astype(int)
    print(list(zip(classes, metrics.f1_score(Y, converted_y_pred, average=None))))
    draw_f1_scores([Y], [converted_y_pred], classes, ["Simple BERT for sequence classification"])
    draw_confusion_matrix(metrics.multilabel_confusion_matrix(Y, converted_y_pred), classes)
    print("Hamming Loss is :", metrics.hamming_loss(Y, converted_y_pred))
    print(metrics.classification_report(Y, converted_y_pred, target_names=classes))
    plot_roc(Y, converted_y_pred, 6)


def print_f1_scores(y_preds, Ys, classes, threshold, model_names):
    for y_pred, Y, model_name in zip(y_preds, Ys, model_names):
        converted_y_pred = (y_pred > threshold).astype(int)
        print("F1-Score for is ", model_name, end=" : ")
        print(list(zip(classes, metrics.f1_score(Y, converted_y_pred, average=None))))


def print_hamming_losses(y_preds, Ys, classes, threshold, model_names):
    for y_pred, Y, model_name in zip(y_preds, Ys, model_names):
        converted_y_pred = (y_pred > threshold).astype(int)
        print("Hamming for", model_name, " : ", metrics.hamming_loss(Y, converted_y_pred))


def print_classification_reports(y_preds, Ys, classes, threshold, model_names):
    for y_pred, Y, model_name in zip(y_preds, Ys, model_names):
        converted_y_pred = (y_pred > threshold).astype(int)
        print("Classification for", model_name, " : \n",
              metrics.classification_report(Y, converted_y_pred, target_names=classes))


def draw_confusion_matrices(y_preds, Ys, classes, threshold, model_names):
    for y_pred, Y, model_name in zip(y_preds, Ys, model_names):
        converted_y_pred = (y_pred > threshold).astype(int)
        print("Confusion matrix for", model_name)
        draw_confusion_matrix(metrics.multilabel_confusion_matrix(Y, converted_y_pred), classes)


def draw_rocs(y_preds, Ys, classes, threshold, model_names):
    for y_pred, Y, model_name in zip(y_preds, Ys, model_names):
        converted_y_pred = (y_pred > threshold).astype(int)
        print("ROC curve for", model_name)
        plot_roc(Y, converted_y_pred, len(classes))


def evaluate_models(y_preds, Ys, classes, threshold, model_names):
    print_f1_scores(y_preds, Ys, classes, threshold, model_names)
    print_hamming_losses(y_preds, Ys, classes, threshold, model_names)
    draw_confusion_matrices(y_preds, Ys, classes, threshold, model_names)
    draw_rocs(y_preds, Ys, classes, threshold, model_names)
    print_classification_reports(y_preds, Ys, classes, threshold, model_names)
