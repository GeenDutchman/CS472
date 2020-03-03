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
            if partition not in self.partitions:
                raise IndexError("Partition " + str(partition) + " not found on this branch.")
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


    def __init__(self):
        # self.root_node = self.Branch(classification, index, partitions)
        self.root_node = None

    def makeBranch(self, classification, index, partitions):
        return self.Branch(classification, index, partitions)
    
    def addBranch(self, child_branch, parent_branch=None, parent_partition=None):
        if parent_branch is None and self.root_node is None:
            self.root_node = child_branch
            return self.root_node
        elif parent_branch is None:
            raise IndexError("Parent not provided while root is defined")
        elif parent_partition is None:
            raise IndexError("Partition not included")
        parent_branch.addChild(parent_partition, child_branch)
        return child_branch

    def makeAddBranch(self, parent_branch: Branch, parent_partition, classification, index, partitions):
        return self.addBranch(self.makeBranch(classification, index, partitions), parent_branch=parent_branch, parent_partition=parent_partition)

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

    def __repr__(self):
        return str(self.root_node)

    

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

if __name__ == "__main__":
    demo()

