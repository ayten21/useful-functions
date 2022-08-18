#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
import zipfile
class HelperFunc:
    def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] 
        n_classes = cm.shape[0]
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        if classes:
            labels = classes
        else:
            labels = np.arange(cm.shape[0])
        ax.set(title="Confusion Matrix",
               xlabel="Predicted label",
               ylabel="True label",
               xticks=np.arange(n_classes),
               yticks=np.arange(n_classes), 
               xticklabels=labels,
               yticklabels=labels)
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        threshold = (cm.max() + cm.min()) / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if norm:
                plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
            else:
                plt.text(j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        if savefig:
            fig.savefig("confusion_matrix.png")


    def calculate_results(y_true, y_pred):
        model_accuracy = accuracy_score(y_true, y_pred) * 100
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        model_results = {"accuracy": model_accuracy,
                         "precision": model_precision,
                         "recall": model_recall,
                         "f1": model_f1}
        return model_results



    def plot_loss_curves(history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        epochs = range(len(history.history['loss']))
        plt.plot(epochs, loss, label='training_loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.figure()
        plt.plot(epochs, accuracy, label='training_accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend();   


    def unzip_data(filename):
        zip_ref = zipfile.ZipFile(filename, "r")
        zip_ref.extractall()
        zip_ref.close()    

    def statistics(data,a):
        print(f"The name of variable: {a}")
        print(f"Data type of variable: {data[a].dtype}")
        print(f"Features: {data[a].shape[0]}")
        data_is_null = data[a].isnull().values.any()
        if data_is_null:
            print(f"Missing values: {data[a].isnull().sum()}")
        else:
            print(f"Null value existance: {data[a].isnull().values.any()}")
        print(f"Unique values: {data[a].nunique()}")
        if data[a].dtype != "0":
            print(f"Min: {int(data[a].min())}")
            print(f"25%: {int(data[a].quantile(q=[.25]).iloc[-1])}")
            print(f"Median: {int(data[a].median())}")
            print(f"75%: {int(data[a].quantile(q=[.75]).iloc[-1])}")
            print(f"Max: {int(data[a].max())}")
            print(f"Mean: {data[a].mean()}")
            print(f"Std dev: {data[a].std()}")
            print(f"Variance: {data[a].var()}")
            print("Percentiles 25%, 50%, 75%, 99%")
            display(data[a].quantile(q=[.25, .5, .75, .99]))     
        else:
            print(f"List of unique values: {data[a].unique()}")


    def histogram(data, a):
        plt.hist(data[a], bins=25)
        plt.title(a, fontsize=10, loc="center")
        plt.xlabel('Relative frequency')
        plt.ylabel('Absolute frequency')
        plt.show()


    def histbox(data, a):
        variable = data[a]
        np.array(variable).mean()
        np.median(variable)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.5, 2)})
        mean=np.array(variable).mean()
        median=np.median(variable)
        sns.boxplot(variable, ax=ax_box)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
        sns.distplot(variable, ax=ax_hist)
        ax_hist.axvline(mean, color='r', linestyle='--')
        ax_hist.axvline(median, color='g', linestyle='-')
        plt.title(x, fontsize=10, loc="right")
        plt.legend({'Mean':mean,'Median':median})
        ax_box.set(xlabel='')
        plt.show()    


    def pie(data, a):
        data[a].value_counts(dropna=False).plot(kind='pie', figsize=(6,5), fontsize=10, autopct='%1.1f%%', startangle=0, legend=True, textprops={'color':"white", 'weight':'bold'})
        val = data[a].value_counts(dropna=False)
        value = pd.DataFrame(val)
        value.rename(columns={a:"Frequency"}, inplace=True)
        value_per = (data[a].value_counts(normalize=True) * 100).round(2)
        value_per = pd.DataFrame(value_per)
        value_per.rename(columns={a:"percent %"}, inplace=True)
        val = pd.concat([value,value_per], axis=1)
        display(val)

    def bar(data, a):
        ax = data[a].value_counts().plot(kind="bar", figsize=(8,6), fontsize=12, color=sns.color_palette("crest"), table=False)
        for i in ax.patches:
            ax.annotate("%.2f" % i.get_height(), (i.get_x() + i.get_width() / 2., i.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        plt.xlabel(a, fontsize=10)
        plt.xticks(rotation=0, horizontalalignment="center")
        plt.ylabel("Absolute values", fontsize=10)
        plt.title(a, fontsize=10, loc="right")


    def scatter(data, x, y, c):
        targets = data[c].unique()
        for target in targets:
            a = data[data[c] == target][x]
            b = data[data[c] == target][y]
            plt.scatter(a, b, label=f" {target}", marker="*")

        plt.xlabel(x, fontsize=10)
        plt.ylabel(y, fontsize=10)
        plt.title("abc", fontsize=10, loc="right")
        plt.legend()
        plt.show()

