import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import BaseDecisionTree, plot_tree
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

dataset = load_iris()

X = dataset.data
y = dataset.target

X, y = shuffle(X, y, random_state=14)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=34)

rf = RandomForestClassifier(n_estimators=100, verbose=0, random_state=14)
result = rf.fit(X_train, y_train)

print(rf.score(X_test, y_test))

# trees_s = last_estimator.estimators_
#
# tree_zero = trees_s[0]
#
# plt.figure(figsize=(10,10))
# plot_tree(tree_zero, feature_names=dataset['feature_names'])
# plt.show()

id_counter = 0
data = []


def export_forest(forest: RandomForestClassifier):
    trees = forest.estimators_
    for tree in trees:
        export_decision_tree(tree)


def export_decision_tree(tree: BaseDecisionTree):
    global id_counter
    global data

    node_count = tree.tree_.node_count

    for node_id in range(node_count):
        children = []

        new_id = node_id + id_counter
        feature_id = None

        predicted_label = None

        if tree.tree_.children_left[node_id] != -1 and tree.tree_.children_right[node_id] != -1:
            left_child = {"child_id": int(tree.tree_.children_left[node_id] + id_counter), "min_value": "-Infinity",
                          "max_value": tree.tree_.threshold[node_id]}
            children.append(left_child)

            right_child = {"child_id": int(tree.tree_.children_right[node_id] + id_counter),
                           "min_value": tree.tree_.threshold[node_id], "max_value": "Infinity"}
            children.append(right_child)

            feature_id = int(tree.tree_.feature[node_id])
        else:
            values = tree.tree_.value[node_id][0]
            if len(values) > 1:  # Classification tree
                max_value = max(values)
                predicted_label = values.tolist().index(max_value)
            else:  # Regression tree
                predicted_label = values[0]

        if len(children) == 0:
            children = None

        class_proportion = tree.tree_.value[node_id][0].tolist()

        node_data = {"node_id": new_id, "children": children, "feature_id": feature_id,
                     "predicted_label": predicted_label, "class_proportions": class_proportion}
        data.append(node_data)

    id_counter = id_counter + node_count


# TODO: export ook de feature names


def export_forest_json(forest: RandomForestClassifier):
    global data
    global id_counter
    data = []
    id_counter = 0

    export_forest(forest)

    if os.path.isfile("./randomForestMap.json"):
        os.remove("./randomForestMap.json")
    with open('./randomForestMap.json', 'w') as f:
        json.dump(data, f)


def export_names(feature_names, target_names):
    if os.path.isfile("./names.json"):
        os.remove("./names.json")
    with open('./names.json', 'w') as f:
        json.dump({"feature_names": feature_names, "target_names": target_names}, f)


def export_dataset(data_X, data_y, column_names, name):
    if os.path.isfile("./" + name):
        os.remove("./" + name)
    df = pd.DataFrame(data=data_X, columns=column_names)
    df['target'] = data_y
    df.to_csv(name, sep=',', index=False)


def export_forests_json(forests):
    global data
    global id_counter
    data = []
    id_counter = 0

    for forest in forests:
        export_forest(forest)

    if os.path.isfile("./randomForestMap.json"):
        os.remove("./randomForestMap.json")
    with open('./randomForestMap.json', 'w') as f:
        json.dump(data, f)


# export_forest_json(rf)
# export_names(dataset["feature_names"], dataset["target_names"].tolist())
# export_dataset(X_train, y_train,  dataset["feature_names"], "trainDataset.csv")
# export_dataset(X_test, y_test,  dataset["feature_names"], "testDataset.csv")

