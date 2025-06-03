import numpy as np
import pickle
import matplotlib.pyplot as plt
from random import shuffle
from typing import (
    List,
    Dict,
    Tuple
)
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd


class MultiClassPerceptron:
    BIAS = 500

    precision, recall, accuracy, fbeta_score = {}, {}, 0, {}

    """
    A Multi-Class Perceptron Model object, with functions for loading feature data, training the algorithm,
    and running analytics on model performance.

    :param  classes           List of categories/classes (match tags in tagged data).
    :param  feature_list      List of features.
    :param  feature_data      Feature Data, in format specified in README, usually imported from feature_data module.
    :param  train_test_ratio  Ratio of data to be used in training vs. testing. Set to 75% by default.
    :param  iterations        Number of iterations to run training data through. Set to 100 by default.
    """

    def __init__(self, classes: List[str], names: List[int], table: List, feature_list: List[str],
                 feature_data: List[Tuple[str, Dict[str, int]]],
                 ttr: float = 1, iterations: int = 100) -> None:

        self.classes = classes
        self.feature_list = feature_list
        self.feature_data = feature_data
        self.iterations = iterations
        self.x_train = self.feature_data[:int(len(self.feature_data) * ttr)]
        self.x_test = self.feature_data[int(len(self.feature_data) * ttr):]
        _, _, self.y_train, self.y_test = train_test_split(table, names,
                                                           test_size=0.1 * (ttr * 10),
                                                           random_state=0)
        self.probability = probability = np.array([[1., 0., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0.8, 0., 0.2],
                                                   [0.2, 0.6, 0.2],
                                                   [0., 1., 0.],
                                                   [0.6, 0., 0.4],
                                                   [0., 1., 0.],
                                                   [0., 1., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0.4, 0., 0.6],
                                                   [0.6, 0., 0.4],
                                                   [0.4, 0., 0.6],
                                                   [0., 1., 0.],
                                                   [1., 0., 0.],
                                                   [0.4, 0., 0.6],
                                                   [1., 0., 0.],
                                                   [0.8, 0., 0.2],
                                                   [1., 0., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0.4, 0., 0.6],
                                                   [1., 0., 0.],
                                                   [0.6, 0., 0.4],
                                                   [0., 1., 0.],
                                                   [0.4, 0.6, 0.],
                                                   [0.6, 0., 0.4],
                                                   [0.4, 0., 0.6],
                                                   [0.6, 0., 0.4],
                                                   [0.8, 0., 0.2],
                                                   [0.4, 0., 0.6],
                                                   [0.2, 0.8, 0.],
                                                   [0., 1., 0.],
                                                   [0.2, 0.6, 0.2],
                                                   [0., 1., 0.],
                                                   [0., 1., 0.],
                                                   [0.4, 0.4, 0.2],
                                                   [0.8, 0., 0.2],
                                                   [0.6, 0., 0.4],
                                                   [1., 0., 0.],
                                                   [0., 1., 0.],
                                                   [0., 1., 0.],
                                                   [0.6, 0., 0.4],
                                                   [0.2, 0.8, 0.],
                                                   [0.6, 0., 0.4],
                                                   [0.6, 0., 0.4],
                                                   [0., 1., 0.],
                                                   [0.4, 0., 0.6],
                                                   [0.6, 0., 0.4],
                                                   [0.2, 0.8, 0.],
                                                   [0.4, 0.6, 0.],
                                                   [1., 0., 0.],
                                                   [0.4, 0., 0.6],
                                                   [0., 1., 0.],
                                                   [1., 0., 0.],
                                                   [1., 0., 0.],
                                                   [0.6, 0., 0.4],
                                                   [0., 1., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0.2, 0.8, 0.],
                                                   [0.6, 0., 0.4],
                                                   [0.8, 0., 0.2],
                                                   [0.4, 0.6, 0.],
                                                   [0.8, 0., 0.2],
                                                   [0.8, 0., 0.2],
                                                   [0.4, 0., 0.6],
                                                   [0., 1., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0., 1., 0.],
                                                   [0.2, 0.6, 0.2],
                                                   [0., 1., 0.],
                                                   [0.4, 0.4, 0.2],
                                                   [0.6, 0.2, 0.2],
                                                   [1., 0., 0.],
                                                   [0.6, 0., 0.4],
                                                   [0.2, 0.8, 0.],
                                                   [0.8, 0., 0.2],
                                                   [1., 0., 0.],
                                                   [0., 1., 0.],
                                                   [1., 0., 0.],
                                                   [0.6, 0., 0.4],
                                                   [0.2, 0.8, 0.],
                                                   [0.4, 0., 0.6],
                                                   [0., 1., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0., 1., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0.8, 0., 0.2],
                                                   [0.2, 0., 0.8],
                                                   [0.2, 0.8, 0.],
                                                   [0., 1., 0.],
                                                   [0., 1., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0.6, 0., 0.4],
                                                   [1., 0., 0.],
                                                   [0.6, 0., 0.4],
                                                   [0., 1., 0.],
                                                   [0.8, 0., 0.2],
                                                   [0.8, 0., 0.2],
                                                   [0., 1., 0.],
                                                   [0., 1., 0.],
                                                   [0., 1., 0.]])

        # Initialize empty weight vectors, with extra BIAS term.
        self.weight_vectors = {c: np.array([0 for _ in range(len(feature_list) + 1)]) for c in self.classes}

    def train(self):
        """
        Train the Multi-Class Perceptron algorithm using the following method (from the README):

        During each iteration of training, the data (formatted as a feature vector) is read in, and the dot
        product is taken with each unique weight vector (which are all initially set to 0). The class that
        yields the highest product is the class to which the data belongs. In the case this class is the
        correct value (matches with the actual category to which the data belongs), nothing happens, and the
        next data point is read in. However, in the case that the predicted value is wrong, the weight vectors a
        re corrected as follows: The feature vector is subtracted from the predicted weight vector, and added to
        the actual (correct) weight vector. This makes sense, as we want to reject the wrong answer, and accept
        the correct one.

        After the final iteration, the final weight vectors should be somewhat stable (it is of importance to
        note that unlike the assumptions of the binary perceptron, there is no guarantee the multi-class
        perceptron will reach a steady state), and the classifier will be ready to be put to use.
        """

        for _ in range(self.iterations):
            for category, feature_dict in self.x_train:
                # Format feature values as a vector, with extra BIAS term.
                feature_list = [feature_dict[k] for k in self.feature_list]
                feature_list.append(MultiClassPerceptron.BIAS)
                feature_vector = np.array(feature_list)

                # Initialize arg_max value, predicted class.
                arg_max, predicted_class = 0, self.classes[0]

                # Multi-Class Decision Rule:
                for c in self.classes:
                    current_activation = np.dot(feature_vector, self.weight_vectors[c])
                    if current_activation >= arg_max:
                        arg_max, predicted_class = current_activation, c

                # Update Rule:
                if not (category == predicted_class):
                    self.weight_vectors[category] += feature_vector
                    self.weight_vectors[predicted_class] -= feature_vector

    def predict(self, obj: Dict[str, int]) -> str:
        """
        Categorize a brand-new, unseen data point based on the existing collected data.

        :param obj: Dict[str, int] -> Features of a new object to categorize.
        :return: str -> Return the predicted category for the data point.
        """

        feature_list = [obj[k] for k in self.feature_list]
        feature_list.append(MultiClassPerceptron.BIAS)
        feature_vector = np.array(feature_list)

        # Initialize arg_max value, predicted class.
        arg_max, predicted_class = 0, self.classes[0]

        # Multi-Class Decision Rule:
        for c in self.classes:
            current_activation = np.dot(feature_vector, self.weight_vectors[c])
            if current_activation >= arg_max:
                arg_max, predicted_class = current_activation, c

        return predicted_class

    def run_analytics(self) -> None:
        """
        Runs analytics on the classifier, returning data on precision, recall, accuracy, as well
        as the fbeta score.

        :return: Prints statistics to screen.
        """

        self.calculate_precision()
        self.calculate_recall()
        self.calculate_fbeta_score()
        self.calculate_accuracy()

    def calculate_precision(self) -> None:
        """
        Calculates the precision of the classifier by running algorithm against test set and comparing
        the output to the actual categorization.
        """

        classes = [f[0] for f in self.x_test]
        correct = {c: sum([1 for d in self.x_test if d[0] == c and d[0] == self.predict(d[1])])
                   for c in classes}
        total = {t: sum([1 for l in self.x_test
                         if l[0] == (m := self.predict(l[1])) and l[0] == t or m == t and l[0] != m])
                 for t in classes}

        print("\nPrecision Statistics:")

        for c in correct:
            if not total[c]: total[c] += 0.000000000000001
            self.precision[c] = (correct[c] * 1.0) / (total[c] * 1.0)
            print(f"{c.upper()} Class Precision -> {self.precision[c]}")

    def calculate_recall(self) -> None:
        """
        Calculates the recall of the classifier by running algorithm against test set and comparing
        the output to the actual categorization.
        """

        classes = [f[0] for f in self.x_test]
        correct = {c: sum([1 for d in self.x_test if d[0] == c and d[0] == self.predict(d[1])])
                   for c in classes}
        total = {t: sum([1 for l in self.x_test if l[0] == t]) for t in classes}

        print("\nRecall Statistics:")

        for c in correct:
            self.recall[c] = (correct[c] * 1.0) / (total[c] * 1.0)
            print(f"{c.upper()} Class Recall -> {self.recall[c]}")

    def calculate_accuracy(self) -> float:
        """
        Calculates the accuracy of the classifier by running algorithm against test set and comparing
        the output to the actual categorization.
        """

        correct, incorrect = 0, 0
        for feature_dict in self.x_test:
            if feature_dict[0] == self.predict(feature_dict[1]):
                correct += 1
            else:
                incorrect += 1

        print(f"\nModel Accuracy: {(acc := (correct * 1.0) / ((correct + incorrect) * 1.0))}")
        return acc

    def calculate_fbeta_score(self) -> None:
        """
        Calculates the fbeta score of the classifier by running algorithm against test set and comparing
        the output to the actual categorization.

        Calculated by taking the harmonic mean of the precision and recall values.
        """

        print("\nF-Beta Scores:")
        for c in self.precision:
            self.fbeta_score[c] = 2 * ((self.precision[c] * self.recall[c] + 0.000000000000001) / (
                    self.precision[c] + self.recall[c] + 0.000000000000001))
            print(f"{c.upper()} Class F-Beta Score -> {self.fbeta_score[c]}")

    def plot_mcp_errors(self, test: List, train: List) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(9):
            ax[0].plot(np.arange(1.0, 10.0, 1.0), test[i], marker='.', label=f"k={i + 1}")
            ax[1].plot(np.arange(1.0, 10.0, 1.0), train[i], marker='.', label=f"k={i + 1}")

        ax[0].legend()
        ax[1].legend()

        fig.supxlabel("Test sizes (%)")
        fig.supylabel("Accuracy")

        fig.suptitle("MCP Training & Testing Accuracy")
        ax[0].set_title("Training accuracy")
        ax[1].set_title("Testing accuracy")
        plt.show()

    def plot_roc(self) -> None:
        fpr, tpr, thresholds = roc_curve(self.y_test, self.probability[:, 1], pos_label=2)
        rp_probs = np.array([np.array([0, 1, 0]) for _ in range(len(self.y_test))])
        r_fpr, r_tpr, thresholds = roc_curve(self.y_test, rp_probs[:, 1], pos_label=2)
        r_auc = roc_auc_score(self.y_test, self.probability, multi_class="ovr")
        rp_auc = roc_auc_score(self.y_test, rp_probs, multi_class="ovr")

        plt.plot(r_fpr, r_tpr, linestyle='--', label=f"Random prediction, AUROC={rp_auc}")
        plt.plot(fpr, tpr, marker='.', label=f"MCP, AUROC={r_auc}")
        plt.title("MCP ROC Plot")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self) -> None:

        cm = confusion_matrix(self.y_test, self.probability)
        df_cm = pd.DataFrame(cm, ["Adelie", "Gentoo", "Chinstrap"], ["Adelie", "Gentoo", "Chinstrap"])
        sn.set(font_scale=2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        plt.show()
