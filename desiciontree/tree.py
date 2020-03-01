import numpy as np

class Tree:

    class Branch():
        def __init__(self, classification, index, partitions):
            self.index = index
            self.classification = classification
            self.partitions = {}
            for part in partitions:
                self.partitions[part] = None

        def addChild(self, partition, branch):
            self.partitions[partition] = branch
            return branch

        def getChild(self, partition):
            return self.partitions[partition]

        def __repr__(self):
            out = "[" + str(self.classification) + " -> "
            for index in self.partitions:
                out = out + str(index)
                out = out + ":" + str(self.partitions[index])
                out = out + " , "
            out = out + "]"
            return out


    def __init__(self, classification, index, partitions):
        self.root_node = self.Branch(classification, index, partitions)

    def newBranch(self, curr_branch: Branch, partition, classification, index, partitions):
        return curr_branch.addChild(partition, self.Branch(classification, index, partitions))

    def _getBranch(self, curr_node: Branch, path, path_index):
        if (len(path) - 1 is path_index):
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

    def __repr__(self):
        return str(self.root_node)

    


tree = Tree("a", 1, ["b", "c"])
print(tree)
# def newBranch(self, curr_branch: Branch, partition, classification, index, partitions):
tree.newBranch(tree.root_node, 'b', 'b', 2, ['c'])
tree.newBranch(tree.getBranch(['a']), 'c', 'c', 3, ['b'])
tree.newBranch(tree.getBranch(['a', 'b']), 'c', 'z', 7, [None])

print(tree)

