from typing import Optional, List, Tuple, Dict, Any
from enum import Enum

import numpy as np
import pandas as pd

from math_utils import entropy_calc

NODES: list = []

class FeatureCol:
    """
    col: int (index column in X)
    name: optional name
    """
    def __init__(self, col: int, name: Optional[str] = None):
        self.col = int(col)
        self.name = None if name is None else str(name)

    def get_col(self) -> int:
        return self.col

    def get_name(self) -> Optional[List[Any]]:
        return self.name

    # Allow equality with another FeatureCol or with an int index
    def __eq__(self, other):
        if isinstance(other, FeatureCol):
            return self.col == other.col
        if isinstance(other, int):
            return self.col == other
        return NotImplemented

    def __ne__(self, other):
        r = self.__eq__(other)
        if r is NotImplemented:
            return NotImplemented
        return not r
        
class Node:
    def __init__(
        self,
        data_ids: np.ndarray,
        node_id: int,
        parent_id: Optional[int] = None,
        depth: int = 0,
        information_gain: float = None,
        entropy: float = None,
        features: Optional[List[FeatureCol]] = None,
    ):
        """
        If parent_id and feature_chosen provided -> create node as child of parent,
        and features will be parent's features minus the chosen one.

        Otherwise (root) provide features list.
        """
        self.node_id: int = node_id # append later, len(NODES) will become index
        self.parent_node_id: int = parent_id
        self.data_ids: np.ndarray = data_ids

        self.is_leaf: bool = False
        self.reason_for_leaf: str = None
        self.prediction_dict = None  # majority class for leaf
        self.certainty = None
        
        # Get depth
        self.depth = depth
        self.entropy = entropy
        self.information_gain = information_gain

        # Get features
        self.features = features

        # Setup chosen feature and children for later
        self.chosen_feature = None
        self.children = {}

    def add_chosen_feature(self, chosen_feature: Optional[FeatureCol]):
        self.chosen_feature = chosen_feature
    
    def add_children(self, value: str, children_id: int):
        self.children[value] = children_id

    def get_prediction(self, y_data: np.ndarray):
        # If the prediction is computed before, skip
        if self.prediction_dict is None:
            # Get y in the class from y_data and node's ids
            y = y_data[self.data_ids]

            # Leaf prediction if pure or empty
            if len(y) == 0:
                self.is_leaf = True
                self.prediction = None
                self.certainty = None
                self.max_occurence = None
                self.total = 0

            else:
                # Predict based on majority
                unique_labels, counts = np.unique(y, return_counts=True)
                
                # Get the max occurence and the total
                self.max_occurence = counts[np.argmax(counts)]
                self.total = np.sum(counts)

                if unique_labels.size == 1:
                    self.is_leaf = True
                    self.prediction = unique_labels[0]
                    self.certainty = 1
                else:
                    # Majority class (used if node becomes leaf later)
                    self.prediction = unique_labels[np.argmax(counts)]
                    self.certainty = counts[np.argmax(counts)] / np.sum(counts)

            # Get the prediction dict
            self.prediction_dict = {
                "prediction": self.prediction, 
                "certainty": self.certainty,
                "max occurence": self.max_occurence,
                "total": self.total
            }
        
        # Return percent or not
        return self.prediction_dict

    def __str__(self):
        return f"Node DT (node {self.node_id}; par. {self.parent_node_id}; dep. {self.depth})"
      
class ID3:
    def __init__(
            self,
            x, y,
            col_names: Optional[List[str]] = None,
            unique_dict: Optional[dict] = None
    ):
        # Get unique values
        feature_num = len(x[0])
        self.full_features = []
        if col_names: # With col_names
            assert len(col_names) == feature_num
            for i in range(feature_num):
                self.full_features.append(FeatureCol(i, col_names[i]))
        else:
            for i in range(feature_num):
                self.full_features.append(FeatureCol(i, None))

        # Get id array of x and y
        data_ids = np.array(range(len(x)))

        # Root node (id = 0) and insert node
        self.current_depth = 0
        self.root = Node(data_ids, node_id = 0, depth = 0, features = self.full_features)
        self.nodes: list[Node] = []
        self.nodes.append(self.root)

        # Input data
        self.x = x
        self.y = y

    def train(self, max_depth: int = 4, min_samples_per_leaf: int = 20):
        # Reset if needed
        self.reset()

        # Get split current
        self._recursion(max_depth, min_samples_per_leaf)

    def traverse_tree(self, X_row: dict, node_id: int = 0):
        """
        Đi xuống cây từ node hiện tại (root hoặc bất kỳ node nào).
        
        Parameters
        ----------
        node : Node
            Node hiện tại (bắt đầu từ root).
        X_row : dict
            Một sample (vd: {"Outlook": "Sunny", "Temperature": "Cool", ...})
        NODES : dict
            Dict chứa tất cả các Node, key là node_id
        
        Returns
        -------
        Node
            Node leaf cuối cùng mà sample đi tới.
        """
        current = self.nodes[node_id]

        while not current.is_leaf:
            # Lấy feature đã chọn ở node hiện tại
            feature = current.chosen_feature
            if feature is None:
                break  # không có feature thì dừng

            value = X_row.get(feature.name, None)  # lấy giá trị của sample theo feature
            if value not in current.children:
                break  # nếu không có nhánh phù hợp thì dừng

            # lấy id con và đi tiếp
            child_id = current.children[value]
            current = self.nodes[child_id]

        return current.get_prediction(self.y)

    def reset(self):
        self.nodes = self.nodes[:1]

    def _recursion(self, max_depth: int, min_samples_per_leaf: int, parent_id: int = 0):
        # Get parent node's info
        parent_node = self.nodes[parent_id]
        data_ids = parent_node.data_ids
        org_sample_num = len(data_ids)

        # Get original entropy (for IG)
        if parent_node.entropy is None:
            self.nodes[parent_id].entropy = self._calc_entropy_of_node()

        # Check condition
        if parent_node.is_leaf:
            return
        elif parent_node.depth == max_depth:
            self.nodes[parent_id].is_leaf = True
            self.nodes[parent_id].reason_for_leaf = "Max Depth Reached"
            return
        elif org_sample_num <= min_samples_per_leaf:
            self.nodes[parent_id].is_leaf = True
            self.nodes[parent_id].reason_for_leaf = "Min Number of Samples Reached"
            return
        elif parent_node.entropy == 0:
            self.nodes[parent_id].is_leaf = True
            self.nodes[parent_id].reason_for_leaf = "All in one class"
            return
        
        # Get children depth
        depth = parent_node.depth + 1

        # Get divisions
        divisions = self._divide_subset(parent_id)

        # Loop and calculate IG for each division
        min_entropy = parent_node.entropy
        min_entropy_per_value = {}
        chosen_feature_id = -1
        for i, feature_div in enumerate(divisions):
            # Get value dict and init feature's entropy
            _, value_dict = feature_div
            feature_entropy_dict = {}
            feature_norm_entropy = 0

            # Loop over unique values of the feature (future branches) and calculate entropy
            for val, mask_data_ids in value_dict.items():
                # Get sample num of the branch
                sample_num = len(mask_data_ids)

                # Get entropy of the branch
                ent, sample_num = self._calc_entropy_of_node(mask_data_ids=mask_data_ids, return_sum=True)
                feature_entropy_dict[val] = ent

                # Calc feature's normalize entropy
                feature_norm_entropy += ent * (sample_num / org_sample_num)
            
            # If feature entropy is lower than min
            if feature_norm_entropy < min_entropy:
                min_entropy = feature_norm_entropy
                min_entropy_per_value = feature_entropy_dict
                chosen_feature_id = i
        
        # Check if there is a legit split and calc Information Gain
        if chosen_feature_id == -1:
            self.nodes[parent_id].is_leaf = True
            self.nodes[parent_id].reason_for_leaf = "No Meaningful Split (IG equal 0 or negative)"
            return
        else:
            # Get best division
            feature_col, value_dict = divisions[chosen_feature_id]
            information_gain = parent_node.entropy - min_entropy

            # Add chosen feature for parent
            chosen_feature = self.full_features[feature_col.get_col()]
            parent_node.add_chosen_feature(chosen_feature)

            # Get features for the children node
            branch_features = [f for f in parent_node.features if f != chosen_feature]

            # Loop over the branches
            for value, mask_data_ids in value_dict.items():
                # Get node_id
                node_id = len(self.nodes)

                # Get node and append into nodes
                node = Node(
                    node_id=node_id,
                    parent_id=parent_id,
                    data_ids=mask_data_ids,
                    depth=depth,
                    information_gain=information_gain,
                    entropy=min_entropy_per_value[value],
                    features=branch_features,
                )
                self.nodes.append(node)

                # Add children into parent's info
                parent_node.add_children(value, children_id=node_id)

                # Recursion
                self._recursion(
                    max_depth=max_depth,
                    min_samples_per_leaf=min_samples_per_leaf,
                    parent_id=node_id
                )

    def _divide_subset(self, node_id: int) -> List[Tuple[FeatureCol, Dict[str, np.ndarray]]]:
        """
        Returns a list of tuples:
          [(feature, {feature_value: mask_id_array, ...}), ...]
        For each available feature, return its split subsets.
        """
        # Get node
        node = self.nodes[node_id]
        if node.is_leaf:
            return []
        
        # Get x
        x = self.x[node.data_ids]

        splits_for_all_features = []
        for feature in node.features:
            col_idx = feature.get_col()
            # compute unique values in this column for current node's x
            # supports string/object arrays or numeric
            vals = np.unique(x[:, col_idx])
            value_dict = {}
            for val in vals:
                # Get x from node's data id
                mask = x[:, col_idx] == val
                child_indices = node.data_ids[mask]
                value_dict[val] = child_indices
            splits_for_all_features.append((feature, value_dict))

        return splits_for_all_features
    
    def _calc_entropy_of_node(
            self,
            mask_data_ids: Optional[np.ndarray] = None,
            return_sum: bool = False,
    ) -> float:
        # Check for validity
        if mask_data_ids is not None:
            y = self.y[mask_data_ids]
        else:
            y = self.y

        # Get percentage of each class in this division
        _, counts = np.unique(y, return_counts=True)
        
        # Get the distribution
        prob_dist = counts / counts.sum()

        # Entropy calc
        if return_sum:
            return entropy_calc(probs=prob_dist), counts.sum()
        else:
            return entropy_calc(probs=prob_dist)
        
    def print_leaves(self):
        # If no branch
        if len(self.nodes) == 0:
            print(f"There is no node.")

        # Loop over the leaves
        for node in self.nodes:
            if node.is_leaf:
                print(node)
                print(f"  Entropy    : {node.entropy}")
                print(f"  Children   : {node.children}")
                print(f"  Prediction : {node.get_prediction(self.y)}")
                print(f"  Leaf reason: {node.reason_for_leaf}\n")

if __name__ == "__main__":
    # Example usage
    x = np.array([
        ['sunny', 'hot'],
        ['sunny', 'hot'],
        ['overcast', 'hot'],
        ['rainy', 'mild'],
        ['rainy', 'cool'],
        ['rainy', 'cool'],
        ['overcast', 'cool'],
        ['sunny', 'mild'],
        ['sunny', 'cool'],
        ['rainy', 'mild'],
    ], dtype=object)

    y = np.array(['no','no','yes','yes','yes','no','yes','no','yes','yes'], dtype=object)

    # Make the instance and train
    id3 = ID3(
        x = x,
        y = y,
        col_names = ["weather", "temp"]
    )
    id3.traiṇ(min_samples_per_leaf=2)

    # Check a branch til leaf
    if False:
        n = id3.nodes[0]
        while True:
            print(n)
            print(f"  Entropy    : {n.entropy}")
            print(f"  Children   : {n.children}")
            print(f"  Prediction : {n.get_prediction(id3.y)}")

            # Print out a branch til leaf - in res.txt
            if n.is_leaf:
                print(f"\nReason for leaf: {n.reason_for_leaf}")
                break
            else:
                children = n.children.items()
                for val, children_id in list(n.children.items())[::-1]:
                    try:
                        n = id3.nodes[children_id]
                        break
                    except:
                        pass
    else:
        # Print out only leaves
        id3.print_leaves()