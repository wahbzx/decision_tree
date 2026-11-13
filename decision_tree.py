import numpy as np
import matplotlib.pyplot as plt
import math


# Data Loading
def create_split_data(file_path, training_ratio):
    '''
    Load file data and split into training and testing data
    according to a specified ratio e.g 0.8 where 80% is training and 20% testing
    equally distributed set of each
    '''
    data = np.loadtxt(file_path)
    rooms = {}
    for i in data:
        r = i[7]
        if r in rooms:
            rooms[r].append(i)
        else:
            rooms[r] = [i]
    train = []
    test = []
    for k in rooms:
        room_data = rooms[k]
        split_point = int(len(room_data)*training_ratio)
        train += room_data[:split_point]
        test += room_data[split_point:]
    return np.array(train), np.array(test)

# Training
            
def get_entropy(data):
    total = len(data)
    labels, counts = np.unique(data[:, -1], return_counts=True)
    entropy = 0
    for c in counts:
        entropy += -c/total * math.log2(c/total)
    return entropy

# Requires sorted data and the column the data was sorted on
def get_candidate_splits(sorted_data, col_sort):
    splits = []
    prev_val = None
    prev_label = None
    for d in sorted_data:
        # TODO: require a list of labels seperately so its more data agnostic
        if prev_val and prev_val != d[col_sort] and prev_label and prev_label != d[7]:
            mid = prev_val + ((d[col_sort] - prev_val) / 2)
            splits.append((mid, d[7]))
        prev_val = d[col_sort]
        prev_label = d[7]
    return splits

def find_split(data):
    '''
    - For each attribute order the values 
        - Find candidate split at decision boundaries
        - find split that gives us most IG
    - Repeat for each attribute and return the attr and split
    - resulting in the highest IG
    '''
    # Entropy for dataset
    h_data = get_entropy(data)
    # Last column is label
    no_attr = data.shape[1] - 1
    best_split = None
    for col in range(0, no_attr):
        sorted_data = data[data[:, col].argsort()]
        splits = get_candidate_splits(sorted_data, col)
        # Max information gain (split_point, Info Gain)
        max_ig = (-1, -1)
        for s in splits:
            filter_left = data[:, col] < s[0]
            filter_right = data[:, col] >= s[0]
    
            left_data = data[filter_left]
            right_data = data[filter_right]
            
            total_entropy = (len(left_data)/len(data)) * get_entropy(left_data) + (len(right_data)/len(data)) * get_entropy(right_data)
            max_ig = max(max_ig, (s[0], h_data - total_entropy), key= lambda x: x[1])
        if best_split and max_ig[1] > best_split[1][1]:
            best_split = (col, max_ig)
        if not best_split:
            best_split = (col, max_ig)
    return best_split


def decision_tree_learning(training_dataset, depth):
    labels, counts = np.unique(training_dataset[:, -1], return_counts=True)
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
        filter_left = training_dataset[:, col] < val
        filter_right = training_dataset[:, col] >= val
        l_data = training_dataset[filter_left]
        r_data = training_dataset[filter_right]
        l_branch, l_depth = decision_tree_learning(l_data, depth+1)
        r_branch, r_depth = decision_tree_learning(r_data, depth+1)
        node["left"] = l_branch
        node["right"] = r_branch
        return (node, max(l_depth, r_depth))

# Evaluation

def get_confusion_matrix(test_pred, test_real):
    conf_mat = np.zeros((4,4), int)
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

    
def evaluate(test_data, tree):
    point = test_data[0]
    test_real = test_data[:, -1]
    test_predictions = []
    for point in test_data:
        node = tree[0]
        while node["attribute"] != "leaf":
            if point[node["attribute"]] < node["value"]:
                node = node["left"]
            else:
                node = node["right"]
        test_predictions.append(node["value"])
    test_predictions = np.array(test_predictions)
    conf_mat = get_confusion_matrix(test_predictions, test_real)
    print(conf_mat)
    return get_accuracy(test_predictions, test_real)

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
    fig, ax = plt.subplots(
        figsize=(leaf_count, max_depth)
    )

    # Draw the tree
    def draw(node):
        
        x, y = positions[id(node)]
        if node["attribute"] == "leaf":
            label = f"Class {int(node['value'])}"
            colour = leaf_colour
        else:
            label = f"X[{node['attribute']}] < {node['value']:.2f}"
            colour = attribute_colour
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=9,
            bbox=dict(
                facecolor=colour,
                edgecolor="black"
            ),
            zorder=2
        )
        # Draw all the arrows between the boxes
        if not node["attribute"] == "leaf":
            for decision, child in [
                ("Yes", node["left"]),
                ("No", node["right"])
            ]:
                child_x, child_y = positions[id(child)]
                ax.annotate(
                    "",
                    xy=(child_x, child_y),
                    xytext=(x, y),
                    arrowprops=dict(
                        arrowstyle="-",
                        color="black",
                    ),
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
    plt.savefig(file_name+".svg", bbox_inches="tight")
    print(f"Saved diagram to {file_name}.svg")



if __name__ == "__main__":
    path = "wifi_db/clean_dataset.txt"
    train, test = create_split_data(path, 0.8)
    tree_clean = decision_tree_learning(train, 0)
    accuracy_clean = evaluate(test, tree_clean)
    plot_tree(tree_clean, "clean")

    path = "wifi_db/noisy_dataset.txt"
    train, test = create_split_data(path, 0.8)
    tree_noisy = decision_tree_learning(train, 0)
    accuracy_noisy = evaluate(test, tree_noisy)

    plot_tree(tree_noisy, "noisy")

    print(f"Clean dataset accuracy: {accuracy_clean*100}%")
    print(f"Noisy dataset accuracy: {accuracy_noisy*100}%")