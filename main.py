import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from perceptron import MultiClassPerceptron
from palmerpenguins import load_penguins
from typing import List, Tuple
import pandas as pd
from typing import List, Dict, Optional

penguins = load_penguins()
penguins.dropna(inplace=True)
penguins = penguins.sample(frac=1)


def island(n: str) -> int:
    """
    Takes an island from the penguins dataset and assigns it
    an integer in-place for identity.
    :param n: str -> island
    :return: int -> island_identifier
    """

    if n == "Torgersen":
        return 1
    elif n == "Dream":
        return 2
    else:
        return 3


def species(n: str) -> int:
    """
    Takes a species from the penguins dataset and assigns it
    an integer in-place for identity.
    :param n: str -> species
    :return: int -> species_identifier
    """

    if n == "Adelie":
        return 1
    elif n == "Gentoo":
        return 2
    else:
        return 3


def transmute_data(table: pd.DataFrame) -> Tuple[List, List]:
    """
    Takes the penguin table and transmutes it into a more linear
    list, allowing for easier access and iteration of data.
    :param table: pd.DataFrame -> penguin_table
    :return: List -> penguin_list
    """

    return [(v[1][1], v[1][2], v[1][3], v[1][4], v[1][5], v[1][6], v[1][7])
            for v in table.iterrows()], [v[1][0] for v in table.iterrows()]


def cross_validate(folds: int = 10) -> Dict[int, float]:

    results = {}
    fold, acc = [], []

    for i in range(1, folds):
        x = MultiClassPerceptron(lifeforms, names, tbl, features, feature_data, ttr=i / 10)
        x.train()
        x.run_analytics()
        # x.plot_roc()
        results[i] = x.calculate_accuracy()
        fold.append(i)
        acc.append(results[i])

    fig, ax = plt.subplots()

    ax.plot(fold, acc, marker='.', label=f"Epoch: {MultiClassPerceptron.BIAS}")
    ax.legend()
    ax.set_title("MCP Cross Validation")

    plt.xlabel("Number of folds")
    plt.ylabel("Accuracy")

    plt.show()

    return results


def test_epoch(cap: int = 2000, split: int = 5) -> Dict[int, float] | None:

    results = {}

    for v in range(cap):

        MultiClassPerceptron.BIAS = v
        x = MultiClassPerceptron(lifeforms, names, tbl, features, feature_data, ttr=split / 10)
        results[v] = x.calculate_accuracy()

    return results



lifeforms = ["Adelie", "Gentoo", "Chinstrap"]
features = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex", "year"]
feature_data = [(v[1][0], {"island": island(v[1][1]), "bill_length_mm": int(v[1][2]),
                           "bill_depth_mm": int(v[1][3]), "flipper_length_mm": int(v[1][4]),
                           "body_mass_g": int(v[1][5]), "sex": 1 if v[1][6] == "male" else 2,
                           "year": int(v[1][7])}) for v in penguins.iterrows()]

penguins["species"] = penguins["species"].apply(species)
penguins["island"] = penguins["island"].apply(island)
penguins["sex"] = penguins["sex"].apply(lambda n: 1 if n == "male" else 2)

tbl, names = transmute_data(penguins)

if __name__ == "__main__":
    # test_epoch()
    cross_validate()
    penguin_classifier = MultiClassPerceptron(lifeforms, names, tbl, features, feature_data, ttr=0.3)
    penguin_classifier.train()
    predicted = penguin_classifier.predict({"island": island("Dream"), "bill_length_mm": 160,
                                            "bill_depth_mm": 30, "flipper_length_mm": 28,
                                            "body_mass_g": 1895, "sex": 1,
                                            "year": 2007})
    print(f"Predicted: {predicted}")
    # penguin_classifier.plot_confusion_matrix()
    penguin_classifier.plot_roc()

# x_train, y_train = np.array([[165], [181], [176], [189], [183], [174], [173]]), [51, 61, 69, 64, 62, 59, 57]
# x_test, y_test = np.array([[175], [157], [161], [193], [169], [172], [178]]), [62, 44, 49, 73, 57, 61, 64]
# model = LinearRegression()
# model.fit(x_train, y_train)
# prediction = model.predict(x_test)
#
# plt.scatter(x_test, y_test, color="black")
# plt.plot(x_test, prediction, color="blue", linewidth=3)
#
# plt.xticks(())
# plt.yticks(())

# y = 70
# fig, ax = plt.subplots()
# plt.scatter(x_test, y_test, color="red")
# plt.plot([160, 190], [y, 70], color="black", linewidth=3)
# y_test.sort()
# residuals = [abs(y-i) for i in y_test]
#
#
# ax.set(xlabel="Mouse weight", ylabel="Mouse size", title="L-Regression")
#
# plt.show()
