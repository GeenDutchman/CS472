import numpy as np

class Tree:

    class Branch():
        serial = 0
        def __init__(self, classification, index, partitions, nan_replace='?Unknown?', friendly_split=None):
            if nan_replace != nan_replace:
                raise ValueError("argument 'nan_replace' must equal itself")
            self.nan_replace = nan_replace
            self.index = index
            self.friendly_split = friendly_split
            self.classification = classification
            self.partitions = {}
            for part in [x for x in partitions if x==x]: # filter out nan because nan != nan
                self.partitions[part] = None
            if len([x for x in partitions if x!=x]) > 0:
                self.partitions[self.nan_replace] = None
            self.serial_num = Tree.Branch.serial
            Tree.Branch.serial = Tree.Branch.serial + 1

        def addChild(self, partition, branch):
            if partition != partition:
                partition = self.nan_replace
            if partition not in self.partitions:
                raise IndexError("Partition " + str(partition) + " not found on this branch.")
            self.partitions[partition] = branch
            return branch

        def getChild(self, partition):
            if partition != partition:
                partition = self.nan_replace
            if partition not in self.partitions:
                return None
            return self.partitions[partition]

        def __repr__(self):
            out = "[" + str(self.classification) + " -> "
            for index in self.partitions:
                out = out + str(index)
                out = out + ":" + str(self.partitions[index])
                out = out + " , "
            out = out + "]"
            return out

        def _graph_(self, indent='\t', class_translator=lambda x: x):
            split_index = "index " + str(self.index)
            split_text = "Split on "
            if self.friendly_split is not None:
                split_text = split_text + "'" + str(self.friendly_split) + "' [" + split_index + "]"
            else:
                split_text = split_text + split_index

            translated = str(class_translator(self.classification))
            if translated != str(self.classification):
                translated = translated + " [" + str(self.classification) + "]"
            out = indent + str(self.serial_num) + ' [label="' + split_text + '.\\n Categorize as:' + translated + '"];\n'
            for index in self.partitions:
                child = self.partitions[index]
                if child is not None:
                    out = out + child._graph_(indent=indent, class_translator=class_translator)
                    out = out + indent + str(self.serial_num) + ' -> ' + str(child.serial_num) + '[label="' + str(index) + '"];\n'
            return out


    def __init__(self):
        # self.root_node = self.Branch(classification, index, partitions)
        self.root_node = None

    def addChildTree(self, otherTree, partition):
        self.root_node.addChild(partition, otherTree.root_node)
        return self

    def makeBranch(self, classification, index, partitions, friendly_split=None):
        return self.Branch(classification, index, partitions, friendly_split=friendly_split)
    
    def addBranch(self, child_branch, parent_branch=None, parent_partition=None):
        if parent_branch is None and self.root_node is None:
            self.root_node = child_branch
            return self
        elif parent_branch is None:
            raise IndexError("Parent not provided while root is defined")
        elif parent_partition is None:
            raise IndexError("Partition not included")
        parent_branch.addChild(parent_partition, child_branch)
        return self

    def makeAddBranch(self, parent_branch: Branch, parent_partition, classification, index, partitions, friendly_split=None):
        return self.addBranch(self.makeBranch(classification, index, partitions, friendly_split=friendly_split), parent_branch=parent_branch, parent_partition=parent_partition)

    def _getBranch(self, curr_node: Branch, path, path_index):
        if ((len(path) - 1 is path_index) or (curr_node is None)):
            result = curr_node
            return result
        else:
            child_node = curr_node.getChild(path[path_index + 1])
            if child_node is not None:
                return self._getBranch(child_node, path, path_index + 1)
            else:
                return None

    def getBranch(self, path):
        return self._getBranch(self.root_node, path, 0)

    def _traverse(self, datapoint, travel_node):
        next_node = travel_node.getChild(datapoint[travel_node.index])
        if next_node is None:
            return travel_node.classification
        else:
            return self._traverse(datapoint, next_node)

    def traverse(self, dataPoint):
        if self.root_node is None:
            raise IndexError("Tree not populated")
        travel_node = self.root_node
        return self._traverse(dataPoint, travel_node)
        

    def __repr__(self):
        return str(self.root_node)

    def graph(self, name='Decision Tree', class_translator=lambda x: x):
        out = "digraph \"" + name + "\" {\n"
        if self.root_node is not None:
            out = out + self.root_node._graph_(class_translator=class_translator)
        out = out + "}\n"
        return out
        

    

def demo():
    tree = Tree() #Tree("a", 1, ["b", "c"])
    tree.makeAddBranch(None, 'a', 'a', 1, ['b', 'c'])
    print(tree)
    # def newBranch(self, curr_branch: Branch, partition, classification, index, partitions):
    tree.makeAddBranch(tree.root_node, 'b', 'b', 2, ['c'])
    tree.makeAddBranch(tree.getBranch(['a']), 'c', 'c', 3, ['b'])
    tree.makeAddBranch(tree.getBranch(['a', 'b']), 'c', 'z', 7, [None])

    print(tree)
    assert('[a -> b:[b -> c:[z -> None:None , ] , ] , c:[c -> b:None , ] , ]' == str(tree))
    print(tree.graph())

if __name__ == "__main__":
    demo()

