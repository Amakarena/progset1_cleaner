import math
import random
import numpy as np
import time
import sys
import matplotlib.pyplot as plt


'''
********************************************************************************
******************************** GRAPH CREATION ********************************
********************************************************************************
'''

# graph class
class MyGraph:

    def __init__(graph, vertices):
        graph.vertices = vertices
        graph.adj_list = [[] for _ in range(vertices)]

    #From Geeks4Geeks implementation (https://www.geeksforgeeks.org/adjacency-list-meaning-definition-in-dsa/)
    #adds edges between vertices
    def gen_edge(graph, v1, v2, w):
        graph.adj_list[v1].append((v2,w))
        graph.adj_list[v2].append((v1,w))

    def show_AdjList(graph):
        for i in range(len(graph.adj_list)):
            print(f"{i}: ", end="")
            for j in graph.adj_list[i]:
                print(f"{{{j[0]}, {j[1]}}} ", end="")
            print()
            

# 1. Complete graph with 0 dimensions (uniform random graph)
def uniform_random(graph):
    if graph.vertices <= 0:
        return []
    
    # Pruning threshold, trial and error final equation
    prune_threshold = math.log2(graph.vertices) / graph.vertices

    for i in range(graph.vertices):
        for j in range(i+1, graph.vertices):
            weight = np.random.rand()
            if weight <= prune_threshold:
                graph.gen_edge(i, j, weight)
    
    return graph.adj_list

# 2. Hypercube graph (edge exists if |a-b| = 2^i)
def hypercube_random(graph):
    if graph.vertices <= 0:
        return []
    
    # Largest k for which 2^k < n, needed b/c we skip by this in for loop below
    max_k = int(math.floor(math.log2(graph.vertices))) 
    
    # For each vertex i, compute neighbor i + 2^k (assuring < n), iteratively
    for i in range (graph.vertices):
        for k in range(max_k + 1):
            j = i + (2**k)
            if j < graph.vertices: # still within constrains
                graph.gen_edge(i, j, np.random.rand())
                
    return graph.adj_list


# 3. 2D Hypercube euclidean distance graph
def hypercube_2D(graph):
    if graph.vertices <= 0:
        return []
    
    coordinates = [(np.random.rand(), 
                    np.random.rand()) for _ in range(graph.vertices)]
    
    # Prune threshold, threshold square root to account for higher dimenstion which adds distance between points
    prune_threshold = math.sqrt(math.log2(graph.vertices) / graph.vertices)
    
    for i in range(graph.vertices):
        i_x, i_y = coordinates[i]
        for j in range(i + 1, graph.vertices):
            j_x, j_y = coordinates[j]
            # Check if coordinates are within 'spatial' threshold
            if (abs(i_x - j_x) > prune_threshold or
                abs(i_y - j_y) > prune_threshold):
                continue
            # Else, find its euclidean dist. and add edge
            dist = math.sqrt((j_x - i_x)**2 + (j_y - i_y)**2)
            graph.gen_edge(i, j, dist)
                                 
    return graph.adj_list
    
    
# 4. 3D Hypercube euclidean distance graph
def hypercube_3D(graph):
    if graph.vertices <= 0:
        return []
    
    coordinates = [(np.random.rand(), 
                    np.random.rand(), 
                    np.random.rand()) for _ in range(graph.vertices)
        ]
    
    # Prune-threshold, the threshold is now cubic root to account for the higher dimension which adds distance between points
    prune_threshold = (math.log2(graph.vertices) / graph.vertices)**(1/3)
    
    for i in range(graph.vertices):
        i_x, i_y, i_z = coordinates[i]
        for j in range(i+1, graph.vertices):
            j_x, j_y, j_z = coordinates[j]
            # Check if coordinates are within 'spatial' threshold
            if (abs(i_x - j_x) > prune_threshold or
            abs(i_y - j_y) > prune_threshold or
            abs(i_z - j_z) > prune_threshold):
                continue
            # Else, find its euclidean dist. and add edge
            dist = math.sqrt((j_x - i_x)**2 + (j_y - i_y)**2 + (j_z - i_z)**2)
            graph.gen_edge(i, j, dist)
    return graph.adj_list


# 5. 4D Hypercube euclidean distance graph
def hypercube_4D(graph):
    if graph.vertices <= 0:
        return []
    
    coordinates = [(np.random.rand(), 
                    np.random.rand(), 
                    np.random.rand(),
                    np.random.rand()) for _ in range(graph.vertices)
        ]
    
    # Prune-threshold, the threshold is now ^1/4 to account for the higher dimension which adds distance between points
    prune_threshold = (math.log2(graph.vertices) / graph.vertices)**(1/4)
    
    for i in range(graph.vertices):
        i_w, i_x, i_y, i_z = coordinates[i]
        for j in range(i+1, graph.vertices):
            j_w, j_x, j_y, j_z = coordinates[j]
            # Check if coordinates are within 'spatial' threshold
            if (abs(i_w - j_w) > prune_threshold or
            abs(i_x - j_x) > prune_threshold or
            abs(i_y - j_y) > prune_threshold or
            abs(i_z - j_z) > prune_threshold):
                continue
            # Else, find its euclidean dist. and add edge
            dist = math.sqrt((j_w - i_w)**2 + (j_x - i_x)**2 + (j_y - i_y)**2 + (j_z - i_z)**2)
            graph.gen_edge(i, j, dist)
           
    return graph.adj_list
    
    
# DFS to check if all nodes visited, else not connected. note: assumes forward/backward relation
def is_connected(adjacency_list):
    
    n = len(adjacency_list)
    if n <= 1:
        return True
    visited = set()
    stack = [0]
    visited.add(0)
    while stack:
        node = stack.pop()
        # adjacency_list[node] is the list of (neighbor, weight) pairs
        for (neighbor, _) in adjacency_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return (len(visited) == n)


'''
********************************************************************************
******************************** MST ALGORITHM *********************************
********************************************************************************
'''

class minHeap:
    def __init__(self, d=4): #generally optimal 4-heap based on slack overflow
        self.heap = []
        # keep track of node index in heap
        self.index_dict = {}
        self.d = d # for implementing a d-heap; base is binary

    # add to heap
    def push(self, node, weight):
        self.heap.append((node, weight))
        self.index_dict[node] = len(self.heap)-1
        # put in correct position in heap
        self.move_up(len(self.heap)-1)

    # remove smallest node from heap
    def pop_node(self):
        # check empty case
        if not self.heap:
            return None
            
        smallest = self.heap[0]
        last = self.heap.pop()

        if self.heap:
            # update index
            self.heap[0] = last
            self.index_dict[last[0]] = 0
            self.move_down(0)
        
        # update dict and return smallest node and its weight
        self.index_dict.pop(smallest[0])
        return smallest

    # to move nodes down the heap to correct position
    def move_down(self, head):
        length = len(self.heap)
        while True:
            # indices of smalles and its children
            smallest = head
            oldest_child = self.d * head + 1
            youngest_child = self.d * head + self.d

            # index movement based on children weight
            for child_i in range(oldest_child, min(youngest_child + 1, length)):
                if self.heap[smallest][1] > self.heap[child_i][1]:
                    smallest = child_i

            #end condition
            if smallest == head:
                break

            # swap nodes and update head
            self.heap[head], self.heap[smallest] = self.heap[smallest], self.heap[head]
            self.index_dict[self.heap[head][0]] = head
            self.index_dict[self.heap[smallest][0]] = smallest
            head = smallest

    # move nodes up
    def move_up(self, child):
        parent = (child - 1) // self.d
        while child > 0 and self.heap[parent][1] > self.heap[child][1]:
            self.heap[parent], self.heap[child] = self.heap[child], self.heap[parent]
            self.index_dict[self.heap[parent][0]] = parent
            self.index_dict[self.heap[child][0]] = child
            # the grasshopper becomes the master
            child = parent
            parent = (child - 1) // self.d

    # update node weight
    def update(self, node, new_weight):
        if node in self.index_dict:
            # old values
            i = self.index_dict[node]
            _ , prev_weight = self.heap[i]
            # actual update
            self.heap[i] = (node, new_weight)

            # new weight, new position in heap
            if prev_weight > new_weight:
                self.move_up(i)
            else:
                self.move_down(i)

# THE mst algo :D
def Prims(graph, d=4): #default 4-ary heap
    vertices = graph.vertices
    # set up arrays/heaps
    dist = [float('inf')] * vertices
    in_visited = [False] * vertices
    visited = minHeap(d)
    to_explore = set(range(vertices))
    prev = [-1] * vertices

    #select random starting node and assign distance and weight
    s = random.randint(0, graph.vertices - 1)
    dist[s] = 0
    visited.push(s,dist[s])

    #explore nodes connected  to s via min edge weight
    while to_explore:
        #remove top
        node = visited.pop_node()
        if node is None: # base case
            continue
        u, min_weight = node

        #handle case if u has been visited already
        if in_visited[u]:
            continue
        
        in_visited[u] = True
        to_explore.remove(u)
        #check adj list and update weights/dists
        for v, w in graph.adj_list[u]:
            if v in to_explore:
                if dist[v] > w:
                    dist[v] = w
                    prev[v] = u
                    if v in visited.index_dict:
                        visited.update(v, w)
                    else:
                        visited.push(v, w)
            
    return dist, prev


'''
********************************************************************************
******************************** MST TESTS *************************************
********************************************************************************
'''
# Helper function to compute the total weight of the MST using the parent list.
def calculate_mst_weight(graph):
    _ , parent = Prims(graph)
    total_weight = 0.0
    for v in range(graph.vertices):
        u = parent[v]
        if u == -1:
            continue  # Root vertex (or not reached in a disconnected graph)
        # Find the edge weight from u to v in the graph's adjacency list.
        for neighbor, weight in graph.adj_list[u]:
            if neighbor == v:
                total_weight += weight
                break
    return total_weight

# Test case 1: Simple triangle graph.
def test_mst_simple_triangle():
    graph = MyGraph(3)
    graph.gen_edge(0, 1, 1)  # Edge weight 1
    graph.gen_edge(1, 2, 2)  # Edge weight 2
    graph.gen_edge(0, 2, 3)  # Edge weight 3 (not used in MST)
    mst_weight = calculate_mst_weight(graph)
    assert abs(mst_weight - 3) < 1e-6, f"Expected MST weight 3, got {mst_weight}"
    print("Simple triangle MST test passed!")

# Test case 2: Graph with two vertices.
def test_mst_two_vertices():
    graph = MyGraph(2)
    graph.gen_edge(0, 1, 10)
    mst_weight = calculate_mst_weight(graph)
    assert abs(mst_weight - 10) < 1e-6, f"Expected MST weight 10, got {mst_weight}"
    print("Two vertices MST test passed!")

# Test case 3: Square graph.
def test_mst_square():
    graph = MyGraph(4)
    graph.gen_edge(0, 1, 1)
    graph.gen_edge(1, 2, 1)
    graph.gen_edge(2, 3, 1)
    graph.gen_edge(0, 3, 10)
    graph.gen_edge(0, 2, 5)
    mst_weight = calculate_mst_weight(graph)
    # Expected MST: edges with weights 1, 1, 1 (total = 3)
    assert abs(mst_weight - 3) < 1e-6, f"Expected MST weight 3, got {mst_weight}"
    print("Square MST test passed!")

# Test case 4: Random uniform graph MST test.
def test_mst_random_uniform():
    n = 10
    graph = MyGraph(n)
    uniform_random(graph)
    if is_connected(graph.adj_list):
        mst_weight = calculate_mst_weight(graph)
        print(f"Random uniform MST weight for n={n}: {mst_weight}")
    else:
        print(f"Random uniform graph for n={n} is disconnected. Skipping MST test.")



'''
********************************************************************************
********************** UNIT TESTS FOR GRAPH GENERATORS *************************
********************************************************************************
'''

# Test for uniform_random
def test_uniform_random():
    # Test n=0: Expect an empty adjacency list
    graph0 = uniform_random(graph = MyGraph(0))
    assert graph0 == [], "Test failed for n=0: expected an empty list (no vertices)."

    # Test n=1: One vertex with an empty edge list.
    graph1 = uniform_random(graph = MyGraph(1))
    assert len(graph1) == 1, "Test failed for n=1: expected one vertex."
    assert graph1[0] == [], "Test failed for n=1: expected no edges."
    
    # Test a few larger values of n
    # (You can adjust or remove these as desired.)
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        g = uniform_random(graph = MyGraph(n))
        if is_connected(g):
            print(f"Uniform_random graph with n={n} is CONNECTED using threshold.")
        else:
            print(f"Uniform random graph with n={n} is DISCONNECTED (rare). Try a better threshold.")

    print("All tests for uniform_random passed!")

def test_hypercube_random():
    """
    Tests the hypercube_uniform function on various values of n.
    Checks that:
      1) The graph is empty for n=0, and trivial for n=1.
      2) For larger n, we see if the graph is connected or not.
    """
    # Test n=0
    g0 = hypercube_random(graph = MyGraph(0))
    assert g0 == [], "Test failed for n=0: expected an empty list."
    
    # Test n=1
    g1 = hypercube_random(graph = MyGraph(1))
    assert len(g1) == 1, "Test failed for n=1: expected one vertex."
    assert g1[0] == [], "Test failed for n=1: expected no edges."

    # Now test some larger n
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
        g = hypercube_random(graph = MyGraph(n))
        if is_connected(g):
            print(f"Hypercube random graph with n={n} is CONNECTED.")
        else:
            print(f"Hypercube random graph with n={n} is DISCONNECTED. Adjust threshold?")
    
    print("All tests for hypercube_uniform passed!")
    
def test_hypercube_2D():
    # Test n=0: Expect an empty adjacency list
    graph0 = hypercube_2D(graph = MyGraph(0))
    assert graph0 == [], "Test failed for n=0: expected an empty list (no vertices)."

    # Test n=1: One vertex with an empty edge list.
    graph1 = hypercube_2D(graph = MyGraph(1))
    assert len(graph1) == 1, "Test failed for n=1: expected one vertex."
    assert graph1[0] == [], "Test failed for n=1: expected no edges."

    # Test a few larger values of n
    # (You can adjust or remove these as desired.)
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        g = hypercube_2D(graph = MyGraph(n))
        if is_connected(g):
            print(f"2D hypercube graph with n={n} is CONNECTED.")
        else:
            print(f"2D hypercube graph with n={n} is DISCONNECTED. Adjust threshold?")

    print("All tests for unit_square_euclidean passed!")

def test_hypercube_3D():
    # Test n=0: Expect an empty adjacency list
    graph0 = hypercube_3D(graph = MyGraph(0))
    assert graph0 == [], "Test failed for n=0: expected an empty list (no vertices)."

    # Test n=1: One vertex with an empty edge list
    graph1 = hypercube_3D(graph = MyGraph(1))
    assert len(graph1) == 1, "Test failed for n=1: expected one vertex."
    assert graph1[0] == [], "Test failed for n=1: expected no edges."

    # Test a few larger values of n
    # You can adjust or remove these values as needed
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        g = hypercube_3D(graph = MyGraph(n))
        if is_connected(g):
            print(f"3D hypercube graph with n={n} is CONNECTED.")
        else:
            print(f"3D hypercube graph with n={n} is DISCONNECTED. Adjust threshold?")

    print("All tests for unit_cube_euclidean passed!")
    
def test_hypercube_4D():
    # Test n=0 => Empty adjacency
    graph0 = hypercube_4D(graph = MyGraph(0))
    assert graph0 == [], "Test failed for n=0: expected an empty list."

    # Test n=1 => One vertex, no edges
    graph1 = hypercube_4D(graph = MyGraph(1))
    assert len(graph1) == 1, "Test failed for n=1: expected one vertex."
    assert graph1[0] == [], "Test failed for n=1: expected no edges."

    # A few larger n
    for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        g = hypercube_4D(graph = MyGraph(n))
        # If you have an is_connected(g) function:
        if is_connected(g):
            print(f"4D hypercube graph with n={n} is CONNECTED.")
        else:
            print(f"4D hypercube graph with n={n} is DISCONNECTED. Adjust threshold?")

    print("All tests for hypercube_euclidean passed!")
    
'''
********************************************************************************
*************************** Running Experiment *********************************
********************************************************************************
'''
# Note: assumes low probability of producing disconnected graphs
def run_experiments(dimension_func, list_of_n, trials=5):
    results = {}
    for n in list_of_n:
        cum_weight = 0.0
        success_count = 0
        while success_count < trials:
            # Construct graph
            g = MyGraph(n)
            # Generate edges using chosen dimension function
            dimension_func(g)
            # Check connecticity 
            if is_connected(g.adj_list):
                # Compute MST weight
                w = calculate_mst_weight(g)
                cum_weight += w
                success_count += 1
            # else, skip and try again
        avg_w = cum_weight / trials
        results[n] = avg_w
        print(f"Dimension func={dimension_func.__name__}, n={n}, avg MST weight={avg_w}")
    return results



'''
********************************************************************************
******************************** THE MAIN **************************************
********************************************************************************
'''

def main():
    flag = int(sys.argv[1])        # flags
    numpoints = int(sys.argv[2])   # num vertices
    numtrials = int(sys.argv[3])   # num trials
    dimension = int(sys.argv[4])   # graph types
    
    # Pick right graph generator function based on dimension given
    if dimension == 0:
        graph_func = uniform_random
    elif dimension == 1:
        graph_func = hypercube_random
    elif dimension == 2:
        graph_func = hypercube_2D
    elif dimension == 3:
        graph_func = hypercube_3D
    elif dimension == 4:
        graph_func = hypercube_4D
            
     # Call run_experiments() with unique n specified
    results = run_experiments(graph_func, [numpoints], numtrials)
    average_weight = results[numpoints]
        
    # output in format given
    print(f"{average_weight} {numpoints} {numtrials} {dimension}")



if __name__ == '__main__':
    # uniform_random(graph = MyGraph(2))
    # hypercube_random(graph = MyGraph(4))
    # hypercube_2D(graph = MyGraph(4))
    # hypercube_3D(graph = MyGraph(4))
    # hypercube_4D(MyGraph(4))
    
    # start = time.process_time()
    # test_uniform_random()
    # print(time.process_time() - start)
    
    # start = time.process_time()
    # test_hypercube_random()
    # print(time.process_time() - start)
    
    # start = time.process_time()
    # test_hypercube_2D()
    # print(time.process_time() - start)
    
    # start = time.process_time()
    # test_hypercube_3D()
    # print(time.process_time() - start)
    
    # start = time.process_time()
    # test_hypercube_4D()
    # print(time.process_time() - start)
    
    # Run MST tests
    # test_mst_simple_triangle()
    # test_mst_two_vertices()
    # test_mst_square()
    # test_mst_random_uniform()
    # main()
    
    '''
    PLOTTING
    '''
    # Example: For dimension=0 (complete graphs), run n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    # (Adjust as needed or as your machine can handle)
    list_of_n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    trials = 5  # run at least 5 times for each n
    # dimension_func = uniform_random  # dimension=0
    
    # For demonstration, let's just pick dimension=0
    results_dim0 = run_experiments(uniform_random, list_of_n, trials)

    # Print results in a "table" (just to standard output)
    print("\nDimension=0, MST Averages:")
    print("n\tAvg MST Weight")
    for n in list_of_n:
        print(f"{n}\t{results_dim0[n]:.6f}")

    # Now you could do a basic plot:
    n_vals = np.array(list_of_n, dtype=float)
    mst_vals = np.array([results_dim0[n] for n in list_of_n], dtype=float)    

    plt.figure()
    plt.plot(n_vals, mst_vals, 'o-', label='Dimension=0')
    plt.xlabel("Number of vertices n")
    plt.ylabel("Average MST Weight")
    plt.title("MST Weight vs. n for dimension=0 (complete graph)")
    plt.legend()
    plt.show()
    
     # (Repeat similarly for dimension=1, dimension=2, etc.)
    results_dim1 = run_experiments(hypercube_random, 
                                   [128, 256, 512, 1024, 2048, 4096, 8192, 
                                    16384, 32768, 65536, 131072, 262144], trials)
    
     # Print results in a "table" (just to standard output)
    print("\nDimension=1, MST Averages:")
    print("n\tAvg MST Weight")
    for n in list_of_n:
        print(f"{n}\t{results_dim1[n]:.6f}")

    # Now you could do a basic plot:
    n_vals = np.array(list_of_n, dtype=float)
    mst_vals = np.array([results_dim1[n] for n in list_of_n], dtype=float)    

    plt.figure()
    plt.plot(n_vals, mst_vals, 'o-', label='Dimension=1')
    plt.xlabel("Number of vertices n")
    plt.ylabel("Average MST Weight")
    plt.title("MST Weight vs. n for dimension=1 (complete graph)")
    plt.legend()
    plt.show()
    
    results_dim2 = run_experiments(uniform_random, list_of_n, trials)

    # Print results in a "table" (just to standard output)
    print("\nDimension=2, MST Averages:")
    print("n\tAvg MST Weight")
    for n in list_of_n:
        print(f"{n}\t{results_dim2[n]:.6f}")

    # Now you could do a basic plot:
    n_vals = np.array(list_of_n, dtype=float)
    mst_vals = np.array([results_dim2[n] for n in list_of_n], dtype=float)    

    plt.figure()
    plt.plot(n_vals, mst_vals, 'o-', label='Dimension=2')
    plt.xlabel("Number of vertices n")
    plt.ylabel("Average MST Weight")
    plt.title("MST Weight vs. n for dimension=2 (complete graph)")
    plt.legend()
    plt.show()
    
    results_dim3 = run_experiments(uniform_random, list_of_n, trials)

    # Print results in a "table" (just to standard output)
    print("\nDimension=3, MST Averages:")
    print("n\tAvg MST Weight")
    for n in list_of_n:
        print(f"{n}\t{results_dim3[n]:.6f}")

    # Now you could do a basic plot:
    n_vals = np.array(list_of_n, dtype=float)
    mst_vals = np.array([results_dim3[n] for n in list_of_n], dtype=float)    

    plt.figure()
    plt.plot(n_vals, mst_vals, 'o-', label='Dimension=3')
    plt.xlabel("Number of vertices n")
    plt.ylabel("Average MST Weight")
    plt.title("MST Weight vs. n for dimension=3 (complete graph)")
    plt.legend()
    plt.show()
    
    results_dim4 = run_experiments(uniform_random, list_of_n, trials)

    # Print results in a "table" (just to standard output)
    print("\nDimension=4, MST Averages:")
    print("n\tAvg MST Weight")
    for n in list_of_n:
        print(f"{n}\t{results_dim4[n]:.6f}")

    # Now you could do a basic plot:
    n_vals = np.array(list_of_n, dtype=float)
    mst_vals = np.array([results_dim4[n] for n in list_of_n], dtype=float)    

    plt.figure()
    plt.plot(n_vals, mst_vals, 'o-', label='Dimension=4')
    plt.xlabel("Number of vertices n")
    plt.ylabel("Average MST Weight")
    plt.title("MST Weight vs. n for dimension=3 (complete graph)")
    plt.legend()
    plt.show()
    
    
    
    
    # Discuss idea that with more time, we could have perhaps explored kruskal's by changing from class into
    # a direct linked list per graph so it could have worked (check if kruskal required manipulation of our code?)