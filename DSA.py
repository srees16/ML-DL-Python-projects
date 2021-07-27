# https://runestone.academy/runestone/books/published/pythonds/index.html
# https://www.programiz.com/dsa
'''
Data Structures:
    Array
    Stack
    Queue
    LinkedList
    Tree: BST
    Heap: Max Heap/Min Heap
    Hash Table
    Trie
    Graph: BFS, DFS

Algorithms:
    Sort:
        bubble, quick, heap, merge, selection, insertion, radix, bucket, shell
    Search:
        linear, binary, jump, interpolation, exponential, Fibonacci seq, Knuth Morris Pratt Pattern Searching
    Recursion

Greedy Algorithms:
    Huffman Code
    Dijkstra's Shortest path for +ve weights
    Prim's Minimum Spanning Tree
    Kruskal's Minimum Spanning Tree
    Ford Fulkerson Max Flow

    Bellman-Ford Shortest path for -ve weights
    Floyd Warshall Shortest path for all pairs
    Edmonds Karp Max Flow
    Kosaraju's Strongly connected Components
    Dynamic Programming

Strings
Memoising
Time complexity analysis

Red-Black tree is a self-balancing BST, each node contains an extra bit for denoting color of the node, either red or black.

Properties of Red-black tree:
    Red/Black Property: Every node is colored, either red or black.
    Root Property: The root is black.
    Leaf Property: Every NIL leaf is black.
    Red Property: If a red node has children, they are always black.
    Depth Property: For each node, any simple path from this node to any of its descendant leaf has the same black-depth (the number of black nodes).

'''

'''''''''''''''''''''''''''''''''''''''''''''''''''Data Structures'''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''Stack'''''''''

class Stack:

    def __init__(self):
         self.items = []

    def is_Empty(self):
        if len(self.items) == 0:
            return True
        return False
        #return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        if not self.is_Empty():
            return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def get_stack(self):
        return self.items

s = Stack()
s.is_Empty()
s.push(4)
s.push('dog')
s.push(True)
s.peek()
s.get_stack()
print(s.size())
print(s.is_Empty())
s.push(8.4)
print(s.pop())
print(s.size())

'''''''''Queue'''''''''

class Queue:

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def get_queue(self):
        return self.items

q = Queue()
q.enqueue(4.1)
q.enqueue(False)
q.enqueue(932)
q.get_queue()
q.isEmpty()
q.dequeue()
q.get_queue()
q.size()

'''''''''Linked List'''''''''
https://www.tutorialspoint.com/python_data_structure/python_linked_lists.htm
https://realpython.com/linked-lists-python/#how-to-insert-a-new-node
# ex: 1
# Node class
class Node:
   
    def __init__(self, data):
        self.data = data
        self.next = None
        
    def __repr__(self):
        return self.data
   
# Linked List class
class LinkedList:

    def __init__(self):
        self.head = None

    def printList(self): # starting from head
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next
    
    def insert_at_begining(self, newdata):
        new_node = Node(newdata) # Update the new nodes next val to existing node
        new_node.next = self.head
        self.head = new_node

# Function to add newnode
    def insert_at_end(self, newdata):
        new_node = Node(newdata)
        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while(last.next):
            last = last.next
        last.next = new_node

# Function to add node
    def insert_in_between(self, middle_node, newdata):
        if middle_node is None:
            print('The mentioned node is absent')
            return
        new_node = Node(newdata)
        new_node.next = middle_node.next
        middle_node.next = new_node

# Function to remove node
    def remove_node(self, remove_key):
        head = self.head
        if head:
            if (head.data == remove_key):
                self.head = head.next
                head = None
                return
        while (head is not None):
            if head.data == remove_key:
                break
            prev = head
            head = head.next
        if (head == None):
            return
        prev.next = head.next
        head = None

    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.next
        return count
    
    def search(self, data):
        current = self.head
        found = False
        while current and found is False:
            if current.data == data:
                found = True
            else:
                current = current.next
        if current is None:
            raise ValueError('Data not in list')
        return current
    
    def __repr__(self):
        node = self.head
        nodes = []
        while node:
            nodes.append(node.data)
            node = node.next
        nodes.append('None')
        return ' -> '.join(nodes)

# Code execution starts here
#if __name__ == '__main__':

# Start with the empty list
llist = LinkedList()
llist.head = Node(1)
second = Node(2)
third = Node(3)
llist.head.next = second
second.next = third
llist.printList()
llist.insert_at_begining('Tue')
llist.insert_at_begining('Wed')
llist.insert_at_begining('Thu')
llist.RemoveNode('Tue')
llist.insert_at_begining('Sun')
llist.insert_at_end('Thu')

llist

'''''''''Tree/Binary Tree'''''''''

#ex: 1
class Node:

    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
        # Compare the new value with the parent node
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data

    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print( self.data),
        if self.right:
            self.right.PrintTree()

root = Node(12)
root.insert(6)
root.insert(14)
root.insert(3)
root.PrintTree()

#ex: 2
class Node:

    def __init__(self, value):
        self.left = None
        self.data = value
        self.right = None

class Tree:

    def createNode(self, data):
        return Node(data)

    def insert(self, node , data):
        if node is None: #if tree is empty, return a root node
            return self.createNode(data)
        if data < node.data: # if data is smaller than parent , insert it into left side
            node.left = self.insert(node.left, data)
        elif data > node.data:
            node.right = self.insert(node.right, data)
        return node

    def search(self, node, data):
        if node is None or node.data == data: # if root is None or root is the search data
            return node
        if node.data < data:
            return self.search(node.right, data)
        else:
            return self.search(node.left, data)

    def deleteNode(self,node,data): # working for only leaf nodes, improve for node
        if node is None: # Check if tree is empty
            return None
        # searching key into BST.
        if data < node.data:
            node.left = self.deleteNode(node.left, data)
        elif data > node.data:
            node.right = self.deleteNode(node.right, data)
        else: # reach to the node that need to delete from BST.
            if node.left is None and node.right is None:
                del node
            if node.left == None:
                temp = node.right
                del node
                return  temp
            elif node.right == None:
                temp = node.left
                del node
                return temp
        return node

    def traverseInorder(self, root):
        if root is not None:
            self.traverseInorder(root.left)
            print(root.data)
            self.traverseInorder(root.right)

    def traversePreorder(self, root):
        if root is not None:
            print(root.data)
            self.traversePreorder(root.left)
            self.traversePreorder(root.right)

    def traversePostorder(self, root):
        if root is not None:
            self.traversePostorder(root.left)
            self.traversePostorder(root.right)
            print(root.data)

def main():
    root = None
    tree = Tree()
    root = tree.insert(root, 10)
    print(root)
    tree.insert(root, 20)
    tree.insert(root, 30)
    tree.insert(root, 40)
    tree.insert(root, 70)
    tree.insert(root, 60)
    tree.insert(root, 80)
    print('Traverse Inorder')
    tree.traverseInorder(root)
    print('Traverse Preorder')
    tree.traversePreorder(root)
    print('Traverse Postorder')
    tree.traversePostorder(root)

if __name__ == '__main__':
    main()

#ex: 3 https://qvault.io/python/binary-search-tree-in-python/
class BSTNode:
    
    def __init__(self, val=None):
        self.left = None
        self.right = None
        self.val = val

    def insert(self, val):
        if not self.val:
            self.val = val
            return

        if self.val == val:
            return

        if val < self.val:
            if self.left:
                self.left.insert(val)
                return
            self.left = BSTNode(val)
            return

        if self.right:
            self.right.insert(val)
            return
        self.right = BSTNode(val)

    def get_min(self):
        current = self
        while current.left is not None:
            current = current.left
        return current.val

    def get_max(self):
        current = self
        while current.right is not None:
            current = current.right
        return current.val

    def delete(self, val):
        if self == None:
            return self
        if val < self.val:
            if self.left:
                self.left = self.left.delete(val)
            return self
        if val > self.val:
            if self.right:
                self.right = self.right.delete(val)
            return self
        if self.right == None:
            return self.left
        if self.left == None:
            return self.right
        min_larger_node = self.right
        while min_larger_node.left:
            min_larger_node = min_larger_node.left
        self.val = min_larger_node.val
        self.right = self.right.delete(min_larger_node.val)
        return self

    def exists(self, val):
        if val == self.val:
            return True

        if val < self.val:
            if self.left == None:
                return False
            return self.left.exists(val)

        if self.right == None:
            return False
        return self.right.exists(val)

    def preorder(self, vals):
        if self.val is not None:
            vals.append(self.val)
        if self.left is not None:
            self.left.preorder(vals)
        if self.right is not None:
            self.right.preorder(vals)
        return vals

    def inorder(self, vals):
        if self.left is not None:
            self.left.inorder(vals)
        if self.val is not None:
            vals.append(self.val)
        if self.right is not None:
            self.right.inorder(vals)
        return vals

    def postorder(self, vals):
        if self.left is not None:
            self.left.postorder(vals)
        if self.right is not None:
            self.right.postorder(vals)
        if self.val is not None:
            vals.append(self.val)
        return vals

def main():
    nums = [12, 6, 18, 19, 21, 11, 3, 5, 4, 24, 18]
    bst = BSTNode()
    for num in nums:
        bst.insert(num)
    print("preorder:")
    print(bst.preorder([]))
    print("#")

    print("postorder:")
    print(bst.postorder([]))
    print("#")

    print("inorder:")
    print(bst.inorder([]))
    print("#")

    nums = [2, 6, 20]
    print("deleting " + str(nums))
    for num in nums:
        bst.delete(num)
    print("#")

    print("4 exists:")
    print(bst.exists(4))
    print("2 exists:")
    print(bst.exists(2))
    print("12 exists:")
    print(bst.exists(12))
    print("18 exists:")
    print(bst.exists(18))

if __name__ == "__main__":
    main()

'''''''''Heap'''''''''
# ex: 1 https://www.programiz.com/dsa/heap-data-structure
def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i],arr[largest] = arr[largest],arr[i]
        heapify(arr, n, largest)

def insert(array, newNum):
    size = len(array)
    if size == 0:
        array.append(newNum)
    else:
        array.append(newNum);
        for i in range((size//2)-1, -1, -1):
            heapify(array, size, i)

def deleteNode(array, num):
    size = len(array)
    i = 0
    for i in range(0, size):
        if num == array[i]:
            break
    array[i], array[size-1] = array[size-1], array[i]
    array.remove(num)
    for i in range((len(array)//2)-1, -1, -1):
        heapify(array, len(array), i)

arr = []
insert(arr, 3)
insert(arr, 4)
insert(arr, 9)
insert(arr, 5)
insert(arr, 2)
print ("Max-Heap array: " + str(arr))
deleteNode(arr, 4)
print("After deleting an element: " + str(arr))

# ex: 2 https://www.section.io/engineering-education/heap-data-structure-python/
class MaxHeap:

    def __init__(self):
        # Initialize a heap using list
        self.heap = []

    def getParentPosition(self, i):
        # The parent is located at floor((i-1)/2)
        return int((i-1)/2)

    def getLeftChildPosition(self, i):
        # The left child is located at 2 * i + 1
        return 2*i+1

    def getRightChildPosition(self, i):
        # The right child is located at 2 * i + 2
        return 2*i+2

    def hasParent(self, i):
        # This function checks if the given node has a parent or not
        return self.getParentPosition(i) < len(self.heap)

    def hasLeftChild(self, i):
        # This function checks if the given node has a left child or not
        return self.getLeftChildPosition(i) < len(self.heap)

    def hasRightChild(self, i):
        # This function checks if the given node has a right child or not
        return self.getRightChildPosition(i) < len(self.heap)

    def insert(self, key):
        self.heap.append(key) # Adds the key to the end of the list
        self.heapify(len(self.heap) - 1) # Re-arranges the heap to maintain the heap property

    def getMax(self):
        return self.heap[0] # Returns the largest value in the heap in O(1) time.

    def heapify(self, i):
        while(self.hasParent(i) and self.heap[i] > self.heap[self.getParentPosition(i)]): # Loops until reaches leaf node
            self.heap[i], self.heap[self.getParentPosition(i)] = self.heap[self.getParentPosition(i)], self.heap[i] # Swap values
            i = self.getParentPosition(i) # Resets new position

    def printHeap(self):
        print(self.heap)

# ex 3: https://www.educative.io/edpresso/heap-implementation-in-python
class MinHeap:
    
    def __init__(self): # On this implementation the heap list is initialized with a value
        self.heap_list = [0]
        self.current_size = 0
 
    def sift_up(self, i): # Moves value up in tree to maintain heap property
        while i // 2 > 0: # While element is not the root or left element
            if self.heap_list[i] < self.heap_list[i // 2]: # if element is less than its parent swap elements
                self.heap_list[i], self.heap_list[i // 2] = self.heap_list[i // 2], self.heap_list[i]
            i = i // 2 # Move index to parent to keep properties
 
    def insert(self, k):
        self.heap_list.append(k)
        self.current_size += 1
        self.sift_up(self.current_size) # Move element to its position from bottom to top
 
    def sift_down(self, i):
        while (i * 2) <= self.current_size: # if the current node has at least one child
            mc = self.min_child(i) # Get the index of the min child of the current node
            if self.heap_list[i] > self.heap_list[mc]: # Swap values of current element if greater than its min child
                self.heap_list[i], self.heap_list[mc] = self.heap_list[mc], self.heap_list[i]
            i = mc
 
    def min_child(self, i):
        if (i * 2)+1 > self.current_size: # if current node has only one child, return index of unique child
            return i * 2
        else:
            if self.heap_list[i * 2] < self.heap_list[(i * 2) + 1]: #if current node has 2 child return index of min child
                return i * 2
            else:
                return (i * 2) + 1
 
    def delete_min(self):
        if len(self.heap_list) == 1: # Equal to 1 since the heap list was initialized with a value
            return 'Empty heap'
        root = self.heap_list[1] # Get root of the heap (The min value of the heap)
        self.heap_list[1] = self.heap_list[self.current_size] # Move the last value of the heap to the root
        *self.heap_list, _ = self.heap_list # Pop the last value since a copy was set on root
        self.current_size -= 1 # Decrease the size of the heap
        self.sift_down(1) # Move down the root (value at index 1) to keep the heap property
        return root # return min value of heap

my_heap = MinHeap()
my_heap.insert(5)
my_heap.insert(6)
my_heap.insert(7)
my_heap.insert(9)
my_heap.insert(13)
my_heap.insert(11)
my_heap.insert(10)
print(my_heap.delete_min())


'''''''''Trie'''''''''
# ex: 1 https://www.geeksforgeeks.org/trie-insert-and-search/
class TrieNode:

    def __init__(self): # Trie node class
        self.children = [None] * 26
        self.isEndOfWord = False # True if node represents end of word
  
class Trie:
    
    def __init__(self):
        self.root = self.getNode()
  
    def getNode(self):
        # Returns new trie node (initialized to NULLs)
        return TrieNode()

    def _charToIndex(self,ch): # Converts key current character into index, use only 'a' through 'z' and lower case
        return ord(ch)-ord('a')

    def insert(self,key):
        # If not present, inserts key into trie. If the key is prefix of trie node, just marks leaf node
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]: # if current character is not present
                pCrawl.children[index] = self.getNode()
            pCrawl = pCrawl.children[index]
        pCrawl.isEndOfWord = True # mark last node as leaf
  
    def search(self, key):
        pCrawl = self.root # Returns true if key presents in trie, else false
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]:
                return False
            pCrawl = pCrawl.children[index]
        return pCrawl != None and pCrawl.isEndOfWord

# driver function
def main():
    # Input keys (use only 'a' through 'z' and lower case)
    keys = ['the', 'a', 'there', 'anaswe', 'any', 'by', 'their']
    output = ['Not present in trie', 'Present in trie']
    t = Trie()
    for key in keys: # Construct trie
        t.insert(key)
    print("{} ---- {}".format("the",output[t.search("the")])) # Search for different keys
    print("{} ---- {}".format("these",output[t.search("these")]))
    print("{} ---- {}".format("their",output[t.search("their")]))
    print("{} ---- {}".format("thaw",output[t.search("thaw")]))

if __name__ == '__main__':
    main()

#ex: 2 https://towardsdatascience.com/implementing-a-trie-data-structure-in-python-in-less-than-100-lines-of-code-a877ea23c1a1

from typing import Tuple

class TrieNode(object):

    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.word_finished = False # Is it the last character of the word
        self.counter = 1 # How many times this character appeared in the addition process

    def add(root, word: str): # Adding a word in the trie structure
        node = root
        for char in word:
            found_in_child = False
            for child in node.children: # Search for the character in the children of the present node
                if child.char == char:
                    child.counter += 1 # if found, increase the counter by 1 to keep track that another word has it as well
                    node = child # And point the node to the child that contains this char
                    found_in_child = True
                    break
            if not found_in_child: # We did not find it so add a new chlid
                new_node = TrieNode(char)
                node.children.append(new_node)
                node = new_node # And then point node to the new child
        node.word_finished = True # Everything finished. Mark it as the end of a word

    def find_prefix(root, prefix: str) -> Tuple[bool, int]:
        node = root
        if not root.children: # If root node has no children, return False, it means empty trie is being searched
            return False, 0
        for char in prefix:
            char_not_found = True
            for child in node.children: # Search through all the children of the present `node`
                if child.char == char:
                    char_not_found = False # We found the char existing in the child
                    node = child # Assign node as the child containing the char and break
                    break
            if char_not_found: # Return False anyway when we did not find a char
                return False, 0
        return True, node.counter # prefix found, and counter of last node indicating how many words have prefix

if __name__ == '__main__':
    root = TrieNode('*')
    add(root, "hackathon")
    add(root, 'hack')
    print(find_prefix(root, 'hac'))
    print(find_prefix(root, 'hack'))
    print(find_prefix(root, 'hackathon'))
    print(find_prefix(root, 'ha'))
    print(find_prefix(root, 'hammer'))

# ex: 3
_end = '_end_'

def make_trie(*words):
    root = dict()
    for word in words:
        current_dict = root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
            current_dict[_end] = _end
            return root

make_trie('foo', 'bar', 'baz', 'barz')

# ex: 4 https://stackoverflow.com/questions/11015320/how-to-create-a-trie-in-python
class Node:

    def __init__(self):
        self.children = [None]*26
        self.isend = False
        
class trie:

    def __init__(self,):
        self.__root = Node()
        
    def __len__(self,):
        return len(self.search_byprefix(''))
    
    def __str__(self):
        ll =  self.search_byprefix('')
        string = ''
        for i in ll:
            string+=i
            string+='\n'
        return string
        
    def chartoint(self,character):
        return ord(character)-ord('a')
    
    def remove(self,string):
        ptr = self.__root
        length = len(string)
        for idx in range(length):
            i = self.chartoint(string[idx])
            if ptr.children[i] is not None:
                ptr = ptr.children[i]
            else:
                raise ValueError("Keyword doesn't exist in trie")
        if ptr.isend is not True:
            raise ValueError("Keyword doesn't exist in trie")
        ptr.isend = False
        return
    
    def insert(self,string):
        ptr = self.__root
        length = len(string)
        for idx in range(length):
            i = self.chartoint(string[idx])
            if ptr.children[i] is not None:
                ptr = ptr.children[i]
            else:
                ptr.children[i] = Node()
                ptr = ptr.children[i]
        ptr.isend = True
        
    def search(self,string):
        ptr = self.__root
        length = len(string)
        for idx in range(length):
            i = self.chartoint(string[idx])
            if ptr.children[i] is not None:
                ptr = ptr.children[i]
            else:
                return False
        if ptr.isend is not True:
            return False
        return True
    
    def __getall(self,ptr,key,key_list):
        if ptr is None:
            key_list.append(key)
            return
        if ptr.isend==True:
            key_list.append(key)
        for i in range(26):
            if ptr.children[i]  is not None:
                self.__getall(ptr.children[i],key+chr(ord('a')+i),key_list)
        
    def search_byprefix(self,key):
        ptr = self.__root
        key_list = []
        length = len(key)
        for idx in range(length):
            i = self.chartoint(key[idx])
            if ptr.children[i] is not None:
                ptr = ptr.children[i]
            else:
                return None
        
        self.__getall(ptr,key,key_list)
        return key_list

t = trie()
t.insert("shubham")
t.insert("shubhi")
t.insert("minhaj")
t.insert("parikshit")
t.insert("pari")
t.insert("shubh")
t.insert("minakshi")

print(t.search("minhaj"))
print(t.search("shubhk"))
print(t.search_byprefix('m'))
print(len(t))
print(t.remove("minhaj"))
print(t)


'''''''''Graphs'''''''''
# ex: 1
# Create the dictionary with graph elements
graph_elements = { 'a' : ['b', 'c'], 'b' : ['a', 'd'], 'c' : ['a', 'd'], 'd' : ['e'], 'e' : ['d']}
print(graph_elements)

class Graph:
    
    def __init__(self, gdict = None):
        if gdict is None:
            gdict = []
        self.gdict = gdict

    # Display vertices, get the keys of the dictionary
    def get_vertices(self):
        return list(self.gdict.keys())
    
    # Display edges
    def edges(self):
        return self.find_edges()

    # Find distinct list of edges
    def find_edges(self):
        edgename = []
        for vrtx in self.gdict:
            for nxtvrtx in self.gdict[vrtx]:
                if {nxtvrtx, vrtx} not in edgename:
                    edgename.append({vrtx, nxtvrtx})
        return edgename
    
    # Add new vertex as a key
    def add_vertex(self, vrtx):
       if vrtx not in self.gdict:
            self.gdict[vrtx] = []

    # Add new edge
    def add_edge(self, edge):
        edge = set(edge)
        (vrtx1, vrtx2) = tuple(edge)
        if vrtx1 in self.gdict:
            self.gdict[vrtx1].append(vrtx2)
        else:
            self.gdict[vrtx1] = [vrtx2]


g = Graph(graph_elements)
print(g.get_vertices())
print(g.edges())
g.add_vertex('f')
print(g.get_vertices())
g.add_edge({'a','e'})
g.add_edge({'a','c'})
print(g.edges())

# ex: 2 https://www.python-course.eu/graphs_python.php
'''Degree of a vertex v in a graph is the number of edges connecting it, with loops counted twice. The degree of a vertex v is denoted deg(v). The maximum degree of a graph G, denoted by Δ(G), and the minimum degree of a graph, denoted by δ(G), are the maximum and minimum degree of its vertices. If all the degrees in a graph are the same, the graph is a regular graph.
In a regular graph, all degrees are the same, and so we can speak of the degree of the graph.
Sum of degrees of all the vertices is equal to the number of edges multiplied by 2, i.e., no of vertices with odd degree has to be even aka handshaking lemma (In any group of people the number of people who have shaken hands with an odd number of other people from the group is even).
The degree sequence of an undirected graph is defined as the sequence of its vertex degrees in a non-increasing order
'''

class Graph(object):

    def __init__(self, graph_dict = None):
        if graph_dict == None:
            graph_dict = {}
        self.graph_dict = graph_dict

    def edges(self, vertice):
        return self.graph_dict[vertice]

    def add_vertex(self, vertex):
        if vertex not in self.graph_dict:
            self.graph_dict[vertex] = []

    def add_edge(self, edge):
        edge = set(edge)
        vertex1, vertex2 = tuple(edge)
        for x, y in [(vertex1, vertex2), (vertex2, vertex1)]:
            if x in self.graph_dict:
                self.graph_dict[x].add(y)
            else:
                self.graph_dict[x] = [y]

    def __generate_edges(self):
        edges = []
        for vertex in self.graph_dict:
            for neighbour in self.graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def all_vertices(self):
        return set(self.graph_dict.keys())

    def all_edges(self):
        return self.__generate_edges()

    def __iter__(self):
        self._iter_obj = iter(self.graph_dict)
        return self._iter_obj
    
    def __next__(self):
        return next(self._iter_obj)

    def __str__(self):
        res = 'vertices: '
        for k in self.graph_dict:
            res += str(k) + ' '
        res += '\nedges: '
        for edge in self.__generate_edges():
            res += str(edge) + ' '
        return res

    def find_path(self, start_vertex, end_vertex, path = None): # find a path from start_vertex to end_vertex in graph
        if path == None:
            path = []
        graph = self.graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex, end_vertex, path)
                if extended_path: 
                    return extended_path
        return None
        
    def find_all_paths(self, start_vertex, end_vertex, path = []): # find all paths from start to end vertex in graph
        graph = self.graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, end_vertex, path)
                for p in extended_paths:
                    paths.append(p)
        return paths

    # Degree of a vertex is no of edges connecting it, i.e., no of adjacent vertices. Loops are counted twice
    def vertex_degree(self, vertex):
        degree =  len(self.graph_dict[vertex]) 
        if vertex in self.graph_dict[vertex]:
            degree += 1
        return degree

    def find_isolated_vertices(self): # returns a list of isolated vertices
        graph = self.graph_dict
        isolated = []
        for vertex in graph:
            print(isolated, vertex)
            if not graph[vertex]:
                isolated += [vertex]
        return isolated
        
    def Delta(self): # maximum degree of the vertices
        max = 0
        for vertex in self.graph_dict:
            vertex_degree = self.vertex_degree(vertex)
            if vertex_degree > max:
                max = vertex_degree
        return max
    
    def degree_sequence(self): # calculates degree sequence
        seq = []
        for vertex in self.graph_dict:
            seq.append(self.vertex_degree(vertex))
        seq.sort(reverse=True)
        return tuple(seq)

g = { 'a' : {'d', 'f'}, 'b' : {'c'}, 'c' : {'b', 'c', 'd', 'e'}, 'd' : {'a', 'c', 'f'}, 'e' : {'c'}, 'f' : {'a', 'd'}}
g1 = { 'a' : {'d', 'f'}, 'b' : {'c'}, 'c' : {'b', 'c', 'd', 'e'}, 'd' : {'a', 'c'}, 'e' : {'c'}, 'f' : {'d'}}

graph = Graph(g)
graph.degree_sequence()
graph1 = Graph(g1)
graph1.degree_sequence()

for vertice in graph:
    print(f"Edges of vertice {vertice}: ", graph.edges(vertice))

graph.add_edge({"ab", "fg"})
graph.add_edge({"xyz", "bla"})

print("Vertices of graph:")
print(graph.all_vertices())
print("Edges of graph:")
print(graph.all_edges())

# calculate the list of all the vertices and the list of all the edges
print("Vertices of graph:")
print(graph.all_vertices())
print("Edges of graph:")
print(graph.all_edges())

# add a vertex and and edge to the graph
print("Add vertex:")
graph.add_vertex("z")

print("Add an edge:")
graph.add_edge({"a", "d"})

print("Vertices of graph:")
print(graph.all_vertices())

print("Edges of graph:")
print(graph.all_edges())

print('Adding an edge {"x","y"} with new vertices:')
graph.add_edge({"x","y"})
print("Vertices of graph:")
print(graph.all_vertices())
print("Edges of graph:")
print(graph.all_edges())

print('The path from vertex "a" to vertex "b":')
path = graph.find_path("a", "b")
print(path)

print('The path from vertex "a" to vertex "f":')
path = graph.find_path("a", "f")
print(path)

print('The path from vertex "c" to vertex "c":')
path = graph.find_path("c", "c")
print(path)

print('All paths from vertex "a" to vertex "b":')
path = graph.find_all_paths("a", "b")
print(path)

print('All paths from vertex "a" to vertex "f":')
path = graph.find_all_paths("a", "f")
print(path)

print('All paths from vertex "c" to vertex "c":')
path = graph.find_all_paths("c", "c")
print(path)

'''''''''''''''''''''''''''''''''''''''''''''''''''Algorithms'''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''Bubble Sort'''''''''
# each pair of adjacent elements is compared and the elements are swapped if they are not in order
def bubblesort(list):
    # Swap the elements to arrange in order
    for iter_num in range(len(list)-1,0,-1):
        for idx in range(iter_num):
            if list[idx]>list[idx+1]:
                temp = list[idx]
                list[idx] = list[idx+1]
                list[idx+1] = temp

listy = [19,2,31,45,6,11,121,27]
bubblesort(listy)
print(listy)

'''''''''Merge Sort'''''''''
# first divides the array into equal halves and then combines them in a sorted manner
def merge_sort(unsorted_list):
    if len(unsorted_list) <= 1:
        return unsorted_list
    # Find the middle point and devide it
    middle = len(unsorted_list) // 2
    left_list = unsorted_list[:middle]
    right_list = unsorted_list[middle:]

    left_list = merge_sort(left_list)
    right_list = merge_sort(right_list)
    return list(merge(left_list, right_list))

# Merge the sorted halves
def merge(left_half, right_half):
    res = []
    while len(left_half) != 0 and len(right_half) != 0:
        if left_half[0] < right_half[0]:
            res.append(left_half[0])
            left_half.remove(left_half[0])
        else:
            res.append(right_half[0])
            right_half.remove(right_half[0])
    if len(left_half) == 0:
        res = res + right_half
    else:
        res = res + left_half
    return res

listy = [64, 34, 25, 12, 22, 11, 90]
print(merge_sort(listy))

'''''''''Insertion Sort'''''''''
# compare the first two elements and sort them by comparing them. Then pick the third element and find its proper position among the previous two sorted elements.
def insertion_sort(alist):
    for i in range(1, len(alist)):
        j = i-1
        nxt_element = alist[i]
        # Compare the current element with next one
        while (alist[j] > nxt_element) and (j >= 0):
            alist[j+1] = alist[j]
            j = j - 1
        alist[j + 1] = nxt_element

lists = [19,2,31,45,30,11,121,27]
insertion_sort(lists)
print(lists)

'''''''''Shell Sort'''''''''
# Sorts elements which are away from each other. sort a large sublist of a given list and go on reducing the size of the list until all elements are sorted. find the gap by equating it to half of the length of the list size and then starts sorting all elements in it. Then keep resetting the gap until the entire list is sorted
def shellSort(input_list):
    gap = len(input_list) // 2
    while gap > 0:
        for i in range(gap, len(input_list)):
            temp = input_list[i]
            j = i
            # Sort the sub list for this gap
            while j >= gap and input_list[j - gap] > temp:
                input_list[j] = input_list[j - gap]
                j = j - gap
            input_list[j] = temp
        # Reduce the gap for the next element
        gap = gap//2

lists = [19,2,31,45,30,11,121,27]
shellSort(lists)
print(lists)

'''''''''Selection Sort'''''''''
# start by finding minimum value in a list and move it to sorted list. Repeat the process for all elements in the unsorted list. The next element entering sorted list is compared with existing elements and placed at its correct position
def selection_sort(input_list):
    for i in range(len(input_list)):
        min_idx = i
        for j in range( i + 1, len(input_list)):
            if input_list[min_idx] > input_list[j]:
                min_idx = j
        # Swap the minimum value with the compared value
        input_list[i], input_list[min_idx] = input_list[min_idx], input_list[i]

l = [19,2,31,45,30,11,121,27]
selection_sort(l)
print(l)

'''''''''Linear Search'''''''''
# sequential search is made over all items one by one. Every item is checked and is returned if a match is found, else search continues till the end
def linear_search(values, search_for):
    search_at = 0
    search_res = False
    # Match the value with each data element
    while search_at < len(values) and search_res is False:
        if values[search_at] == search_for:
            search_res = True
        else:
            search_at = search_at + 1
    return search_res

l = [64, 34, 25, 12, 22, 11, 90]
print(linear_search(l, 12))
print(linear_search(l, 91))

'''''''''Interpolation Search'''''''''
# works on the probing position of a value. data collection should be in a sorted form and equally distributed. Initially, probe position is the position of the middle most item of the collection. If a match occurs, then the index of the item is returned. If the middle item is greater than the item, then the probe position is again calculated in the sub-array to the right of the middle item. Otherwise, the item is searched in the subarray to the left of the middle item. This process continues on the sub-array as well until the size of subarray reduces to zero.
def intpolsearch(values, x):
    idx0 = 0
    idxn = (len(values) - 1)
    while idx0 <= idxn and x >= values[idx0] and x <= values[idxn]:
        # Find the mid point
        mid = idx0 + int(((float(idxn - idx0)/(values[idxn] - values[idx0])) * (x - values[idx0])))
        # Compare the value at mid point with search value
        if values[mid] == x:
            return 'Found ' + str(x) + ' at index ' + str(mid)
        if values[mid] < x:
            idx0 = mid + 1
    return 'Searched element not in the list'

l = [2, 6, 11, 19, 27, 31, 45, 121]
print(intpolsearch(l, 2))

    
'''''''''''''''''''''''''''''''''Time Complexity Analysis'''''''''''''''''''''''''''''''''

'''How to calculate Big O time complexity?
1. drop constants
2. add up all time complexity terms in a function to calculate worse case
3. drop non dominant terms. ex: in O(n**2 + n), n can be dropped and n**2 becomes worst case time complexity
4. in case of different input arrays, Big O calculates as O(a*b), where a & b are lengths of arrays A, B

Use timeit to find time taken for executing a function

Constant O(1)
Linear O(n)
Logarithmic O(log n)
Log Linear O(n log n)
Quadratic O(n**2)
Cubic O(n**3)
Exponential O(2**n)

A logarithmic function is opposite of exponential function. When something grows exponentially, it’s being multiplied. When something grows logarithmically, it is being divided.

'''

my_lis = [(4, 3), (1, 2), (6, 1)]
w = [10, 9, 6, 5, 3, 8]


'''Constant O(1)'''
def first_element(ls):
    return ls[0]

first_element(my_lis)

'''Linear O(n)'''
def squares(li):
    for i in li:
        print(i)

squares(w)

def prints(li): # O(n + n) = O(2n) = O(n), as we drop constants, and repetition of operations doesnt count as time complexity
    for i in li:
        print(i)
    for i in li:
        print(i)

prints(w) # not O(n**2) since its linear

'''Quadratic O(n**2)'''
def quadratic(ls):
    for i in ls:
        for j in ls:
            print(i, j)

quadratic(w)

'''Logarthmic O(log n)'''

# What power do we raise 3 to to get 9? i.e., 3^x == 9
# log3(9) == 2
# if there are 7 list items, it takes one operation to execute since it runs in log base 2 of n time
# to search an element in a list of 7 elements, no of operations it takes to run is log base 2 of input size i.e., 7. In this case runtime is log2(7) or ~3 operations. Binary search uses log

def logs(x):
    while x > 0: # O(log(n))
        y = 2 + 2
        x = x // 2
        return x

logs(10)

'''Big O of a given function'''
def mixy(li):
    print(li[0]) # O(1)

    mids = len(li)//2

    for i in li[:mids]:
        print(i) # O(n)

    for x in range(10): # O(1)
        print('Yes!')

mixy(w) # Big O of above function is O(1 + n + 1) = O(n)

def matching(ls, inputs):
    for i in ls:
        if i == inputs:
            return True
        return False

matching(w, 10) # best case is O(1)
matching(w, 11) # worst case is O(n)

'''Big O plot'''
def m1():
    l = []
    for i in range(10000):
        l += [i]

def m2():
    l = []
    for i in range(10000):
        l.append(i)

def m3():
    l = [n for n in range(10000)]

def m4():
    l = list(range(10000))

timeit m1()
timeit m2()
timeit m3()
timeit m4()

'''Example'''

n = np.linspace(1, 10, 1000)
labels = ['Constant O(1)', 'Logarithmic O(log n)', 'Linear O(n)', 'Log Linear O(n log n)', 'Quadratic O(n**2)', 'Cubic O(n**3)', 'Exponential O(2**n)']

big_o = [np.ones(n.shape), np.log(n), n, n * np.log(n),n ** 2, n ** 3, 2 ** n]

plt.figure(figsize = (12, 10))
plt.ylim(0, 100)
for i in range(len(big_o)):
    plt.plot(n, big_o[i], label = labels[i])
    plt.xlabel('Time Complexities')
    plt.ylabel('Time')
    plt.legend()

'''Sort algorithms'''
#https://www.youtube.com/playlist?list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12
#https://www.youtube.com/playlist?list=PLzgPDYo_3xunyLTJlmoH8IAUvet4-Ka0y


'''
stack based problems
https://www.geeksforgeeks.org/stack-data-structure/?ref=ghm#standard
https://www.fullstack.cafe/interview-questions/stacks
https://medium.com/techie-delight/stack-data-structure-practice-problems-and-interview-questions-9f08a35a7f19
https://www.techiedelight.com/stack-interview-questions/
'''

# Given a string of brackets, return if balanced. Ex: given ([])[]({}) return true, for "([)]" or "((()" return false
    1. scan from left to right
    2. if current char is opening bracket, push it to stack
    3. if current char is closing bracket & top of stack opening is of same type, then pop
    4. should end with empty list if balanced brackets

def bracker_balancer(strs):
    my_stack = []
    opens = ['(', '[', '{']
    for i in strs:
        if i in opens:
            my_stack.append(i)
        closing = my_stack.pop()
        if i == 



# ex: 1 https://www.geeksforgeeks.org/check-for-balanced-parentheses-in-an-expression/?ref=rp
def bracket_checker(st):
    my_stack = []
    opens = ['(', '{', '[']
    for i in st:
        if i in opens:
            my_stack.append(i)
        else:
            if not my_stack:
                return False
            current_char = my_stack.pop()
            if current_char == '(':
                if i != ')':
                    return False
            if current_char == '{':
                if i != '}':
                    return False
            if current_char == '[':
                if i != ']':
                    return False
    if my_stack:
        return False
    return True

bracks = '([])[]({})'
bracks1 = '([)]'
bracks2 = '((()'

bracket_checker(bracks)

# ex:2 https://www.geeksforgeeks.org/check-for-balanced-parenthesis-without-using-stack/?ref=rp
def findClosing(c):
    if c == '(':
        return ')'
    elif c == '{':
        return '}'
    elif c == '[':
        return ']'
    return -1

# function to check if parenthesis are balanced
def check(my_str):
    n = len(my_str)
    # Base cases
    if n == 0:
        return True
    if n == 1:
        return False
    if my_str[0] == ')' or my_str[0] == '}' or my_str[0] == ']':
        return False
    # search for closing bracket for first opening bracket
    closing = findClosing(my_str[0])
    # count is used to handle cases like ((())) to consider matching closing bracket
    i = -1
    count = 0
    for i in range(1, n):
        if my_str[i] == my_str[0]:
            count += 1
        if my_str[i] == closing:
            if count == 0:
                break
            count -= 1
    # if closing bracket is not found
    if i == n:
        return False
    # if closing bracket was next to open
    if i == 1:
        return check(my_str[2:])
    # if closing bracket was somewhere in middle
    return check(my_str[1:]) and check(my_str[i + 1:])

if check(bracks):
    print('Balanced')
else:
    print('Not Balanced')

# Find duplicate parenthesis in an expression https://www.techiedelight.com/find-duplicate-parenthesis-expression/
# Find all strings of a given length containing balanced parentheses https://www.techiedelight.com/find-strings-given-length-containing-balanced-parentheses/
