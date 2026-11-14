import os

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import math

# Depending on the format of the data, its better to write a custom data loader
def load_wifi(file_path):
    data = np.loadtxt(file_path)
    x = data[:, 0:-1]
    y = data[:, -1]
    classes, y = np.unique(y, return_inverse=True)
    return x,y,classes

def load_iris(file_path):
    x = np.loadtxt('iris.data', delimiter=',', usecols=(0, 1, 2, 3))
    y = np.loadtxt('iris.data', delimiter=',', usecols=4, dtype=str)
    classes, y = np.unique(y, return_inverse=True)
    return x,y,classes
    

def create_split_data(x,y, training_ratio):
    assert x.shape[0] == y.shape[0]
    rooms = {}
    for i in range(0,len(y)):
        r = y[i]
        instance = x[i]
        if r in rooms:
            rooms[r].append(instance)
        else:
            rooms[r] = [instance]
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for k in rooms:
        room_data = rooms[k]
        room_labels = [k for i in range(len(room_data))]
        split_point = int(len(room_data)*training_ratio)
        train_x += room_data[:split_point]
        train_y += room_labels[:split_point]
        test_x += room_data[split_point:]
        test_y += room_labels[split_point:]
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def get_entropy(train_y):
    total = len(train_y)
    labels, counts = np.unique(train_y, return_counts=True)
    entropy = 0
    for c in counts:
        entropy += -c/total * math.log2(c/total)
    return entropy


def get_candidate_splits(sorted_x, sorted_y):
    splits = []
    prev_val = None
    prev_label = None
    for i in range(len(sorted_x)):
        feature = sorted_x[i]
        label = sorted_y[i]
        if (prev_val is not None and prev_val != feature) and (prev_label is not None and prev_label != label):
            mid = prev_val + ((feature - prev_val) / 2)
            splits.append((mid, label, i))
        prev_val = feature
        prev_label = label
    return splits

def find_split(data):
    '''
    - For each attribute order the values 
        - Find candidate split at decision boundaries
        - find split that gives us most IG
    - Repeat for each attribute and return the attr and split
    - resulting in the highest IG
    '''
    train_x, train_y = data
    # Entropy for dataset
    h_data = get_entropy(train_y)
    no_attr = train_x.shape[1]
    best_split = None
    for col in range(no_attr):
        indices = np.argsort(train_x[:, col])
        sorted_x = train_x[indices,col]
        sorted_y = train_y[indices]
        splits = get_candidate_splits(sorted_x, sorted_y)
        # Max information gain (split_point, Info Gain)
        max_ig = (-1, -1)
        for s in splits:
            split_index = s[2]
            left_train_y = sorted_y[:split_index]
            right_train_y = sorted_y[split_index:]
            total_entropy = (len(left_train_y)/len(train_y)) * get_entropy(left_train_y) + (len(right_train_y)/len(train_y)) * get_entropy(right_train_y)
            max_ig = max(max_ig, (s[0], h_data - total_entropy), key= lambda x: x[1])
        if best_split and max_ig[1] > best_split[1][1]:
            best_split = (col, max_ig)
        # If best_split has not been initialised yet, set it
        if not best_split:
            best_split = (col, max_ig)
    return best_split


def decision_tree_learning(training_dataset, depth):
    train_x, train_y = training_dataset
    labels, counts = np.unique(train_y, return_counts=True)
    col, max_ig = find_split(training_dataset)
    if len(np.unique(labels)) == 1:
        node = {"attribute":"leaf", "value":labels[0], "left":"", "right":""}
        return (node, depth)
    # Edge case where we couldn't find an appropriate split candidate (due to noise)
    elif max_ig == (-1,-1):
        common = max(zip(labels,counts), key=lambda x: x[1])
        node = {"attribute":"leaf", "value":common[0], "left":"", "right":""}
        return (node, depth)
    else:
        val = max_ig[0]
        node = {"attribute":col, "value":val, "left":"", "right":""}
        l_data_x = []
        l_data_y = []
        r_data_x = []
        r_data_y = []
        for i in range(0, len(train_x)):
            if train_x[i, col] < val:
                l_data_x.append(train_x[i])
                l_data_y.append(train_y[i])
            else:
                r_data_x.append(train_x[i])
                r_data_y.append(train_y[i])
        l_branch, l_depth = decision_tree_learning((np.array(l_data_x), np.array(l_data_y)), depth+1)
        r_branch, r_depth = decision_tree_learning((np.array(r_data_x), np.array(r_data_y)), depth+1)
        node["left"] = l_branch
        node["right"] = r_branch
        return (node, max(l_depth, r_depth))

# Evaluation

def get_confusion_matrix(test_pred, test_real):
    dim = len(np.unique(test_real))
    conf_mat = np.zeros((dim,dim), int)
    pred_map = {}
    for i in range(0,len(test_real)):
        real = test_real[i]
        pred = test_pred[i]
        conf_mat[int(real-1), int(pred-1)] += 1
    return conf_mat
    
    
def get_accuracy(test_predictions, test_real):
    assert(test_predictions.shape == test_real.shape)
    correct = 0
    for i in range(len(test_predictions)):
        if test_predictions[i] == test_real[i]:
            correct += 1
    return correct/len(test_predictions)

    
def evaluate(test_x, test_y, tree):
    test_pred = []
    for i in range(len(test_x)):
        node = tree[0]
        while node["attribute"] != "leaf":
            if test_x[i, node["attribute"]] < node["value"]:
                node = node["left"]
            else:
                node = node["right"]
        test_pred.append(node["value"])
    test_pred = np.array(test_pred)
    conf_mat = get_confusion_matrix(test_pred, test_y)
    # print(f"Confusion Matrix: \n{conf_mat}")
    return get_accuracy(test_pred, test_y)

# Draw tree diagram

def plot_tree(result, file_name="decision_tree"):
    tree, max_depth = result
    positions = {}
    leaf_count = 0
    leaf_colour = "springgreen"
    attribute_colour = "peru"

    # Calculate the positions of each node
    def calc_pos(node, depth=0, leaf_count=0):
        if node["attribute"] == "leaf":
            x = leaf_count
            leaf_count += 1
        else:
            left_x, leaf_count = calc_pos(node["left"], depth + 1, leaf_count)
            right_x, leaf_count = calc_pos(node["right"], depth + 1, leaf_count)
            x = (left_x + right_x) / 2
        positions[id(node)] = (x, -depth)
        return x, leaf_count

    _, leaf_count = calc_pos(tree)
    print(f"{leaf_count} Leaves")
    fig, ax = plt.subplots(figsize=(leaf_count, max_depth))

    # Draw the tree
    def draw(node):
        x, y = positions[id(node)]
        if node["attribute"] == "leaf":
            label = f"Class {node['value']}"
            colour = leaf_colour
        else:
            label = f"X[{node['attribute']}] < {node['value']:.2f}"
            colour = attribute_colour
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=9,
            bbox=dict(facecolor=colour, edgecolor="black"),
            zorder=2
        )
        # Draw all the arrows between the boxes
        if not node["attribute"] == "leaf":
            for decision, child in [("Yes", node["left"]),("No", node["right"])]:
                child_x, child_y = positions[id(child)]
                ax.annotate(
                    "",
                    xy=(child_x, child_y),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-",color="black"),
                    zorder=1
                )
                ax.text(
                    (x + child_x) / 2,
                    (y + child_y) / 2,
                    decision,
                    ha="right"
                )
                draw(child)

    draw(tree)

    ax.set_xlim(-1, leaf_count)
    ax.set_ylim(-max_depth - 1, 1)
    ax.axis("off")

    plt.tight_layout()
    folder = "tree_diagrams/"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder+file_name+".svg", bbox_inches="tight")


# Stratified K-fold cross validation
# 1. Split intro stratas
# 2. Shuffle the instances within each strata
# 3. Split each strata into k partitions
# 4. Rotate the held out test fold across all k iterations
# 5. Use the remaining k-1 folds for training across all k iterations
def k_fold_cross(x,y,seed):
    k=10
    rg = default_rng(seed)
    # Split intro stratas
    stratas = {}
    for i in range(0,len(y)):
        s = y[i]
        instance = x[i]
        if s in stratas:
            stratas[s].append(instance)
        else:
            stratas[s] = [instance]
    # Shuffle the instances within each strata
    for s in stratas:
        folds = np.array_split(rg.permutation(stratas[s]), k)
        stratas[s] = folds
    # every fold gets its own time to shine as the test fold
    accuracies = []
    for i in range(k):
        train_list_x = []
        train_list_y = []
        test_list_x = []
        test_list_y = []
        for s in stratas:
            # Extract test fold
            test_fold = stratas[s][i]
            test_list_x.append(test_fold)
            test_list_y.append(np.full(len(test_fold), s))
            # Extract training folds
            for j, fold in enumerate(stratas[s]):
                if j != i:
                    train_list_x.append(fold)
                    train_list_y.append(np.full(len(fold), s))
        
        # Convert to np array by stacking them in row-major format           
        train_x = np.vstack(train_list_x)
        test_x = np.vstack(test_list_x)
        train_y = np.concatenate(train_list_y)
        test_y = np.concatenate(test_list_y)
        # Train using this split and evaluate tree
        training_dataset = (train_x, train_y)
        tree = decision_tree_learning(training_dataset, 0)
        accuracy = evaluate(test_x, test_y, tree)
        accuracies.append(accuracy)
    
    print(f"Mean accuracy: {(np.mean(accuracies)*100):.2f}%")
    print(f"Std deviation across folds: ± {(np.std(accuracies)*100):.2f}%")
    return accuracies

if __name__ == "__main__":
    seed = 1337
    path = "wifi_db/clean_dataset.txt"
    x,y,classes = load_wifi(path)
    print("Results for clean dataset")
    accuracies = k_fold_cross(x, y, seed)

    print("\n")

    path = "wifi_db/noisy_dataset.txt"
    x,y,classes = load_wifi(path)
    seed = 1337
    print("Results for noisy dataset")
    accuracies = k_fold_cross(x, y, seed)
