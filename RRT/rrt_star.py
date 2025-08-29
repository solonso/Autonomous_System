import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Point import Point
import sys
import copy
import argparse
from rrt_recorder import add_recording_args, setup_recording, plot_with_recording

def plot(gridmap,start,goal,tree,path,show_vertcies=False):
    """
    Visualizes the grid map, tree structure, and the path from start to goal, with optional vertex labeling.

    Parameters:
    - gridmap (2D array): The grid map representing the environment, where `0` indicates free spaces and non-zero values represent obstacles.
    - start (tuple): The (row, col) coordinates of the start position.
    - goal (tuple): The (row, col) coordinates of the goal position.
    - tree (dict): A dictionary where keys are parent vertices and values are child vertices, representing the tree structure.
    - path (list): A list of vertices (nodes) representing the path from start to goal.
    - show_vertcies (bool): If True, displays vertices with their indices. Default is False.

    Returns:
    - matplotlib.figure.Figure: The figure object for further manipulation.

    Notes:
    - The grid map is visualized using `matshow`, providing a background of the environment.
    - Tree edges are drawn as white lines between parent and child nodes.
    - The given path is shown as a series of red lines connecting the nodes in the path.
    - Start and goal positions are marked with distinct symbols (`r*` for start and `g*` for goal), and labeled in the legend.
    - If `show_vertcies` is True, each vertex in the tree is marked with a green plus sign (`+`), and its index is displayed next to it.
    - A legend is included to distinguish between the start and goal positions.
    """

    fig = plt.figure(figsize=(10, 10))
    plt.matshow(gridmap, fignum=0)

    # Plot vertcies
    if show_vertcies:
        for i,v in enumerate(tree.keys()):
            plt.plot(v.y, v.x, "+g")
            plt.text(v.y, v.x, i, fontsize=14, color="w")

    # Plot Edges
    for parent,child in tree.items():
        if child != None:
            plt.plot(
                [parent.y,child.y],
                [parent.x,child.x],
                '-w'
            )
    # Plot given Path
    for i in range(1, len(path)):
        plt.plot(
            [path[i - 1].y, path[i].y],
            [path[i - 1].x, path[i].x],
            "r",
        )
    # Start
    plt.plot(start[1], start[0], "r*",markersize=10,label='Start')
    # Goal
    plt.plot(goal[1] ,goal[0], "g*",markersize=10,label='Goal')
    plt.legend()
    
    return fig


class RRTStar:

    """
    A class to implement the RRT* (Rapidly-Exploring Random Tree Star) algorithm for path planning in a grid-based environment.
    RRT* extends RRT by optimizing paths to minimize cost.

    Attributes:
        gridmap (np.ndarray): 2D numpy array representing the grid map of the environment. 
                              Cells with value 0 are valid configurations; others are obstacles.
        max_iter (int): The maximum number of iterations to grow the RRT*.
        dq (float): Step size for extending the tree.
        p (float): Probability of selecting the goal as the target in a given iteration (goal bias).
        max_search_distance (float): Maximum distance to search for nearby nodes during the rewire step.
        start (Point): Starting point for the RRT* algorithm, initialized as a Point object.
        goal (Point): Goal point for the RRT* algorithm, initialized as a Point object.
        valid_rows (np.ndarray): Array of row indices for valid configurations in the grid map.
        valid_cols (np.ndarray): Array of column indices for valid configurations in the grid map.

    """
    def __init__(self,gridmap,max_iter,dq,p,max_search_distance,start,goal):
        self.gridmap = gridmap
        self.max_iter = max_iter
        self.dq = dq
        self.p = p
        self.max_search_distance = max_search_distance
        self.start = Point(start[0],start[1])
        self.goal = Point(goal[0],goal[1])
        self.valid_rows,self.valid_cols = np.where(self.gridmap == 0) # Get indcies for valid configurations


    def sample_random_point(self,valid_rows,valid_cols,p):
            
        """
        Samples a random point from the set of valid grid locations, with a probability of returning the goal point.

        Parameters:
        - valid_rows (array-like): A list or array of valid row indices for free spaces in the grid.
        - valid_cols (array-like): A list or array of valid column indices for free spaces in the grid.
        - p (float): The probability of returning the goal point instead of a randomly chosen point. Must be between 0 and 1.

        Returns:
        - Point: A randomly sampled point from the set of valid grid locations. The point will either be the goal (with probability `p`) or a random valid point (with probability `1 - p`).

        Notes:
        - The function selects a random index from the `valid_rows` and `valid_cols` arrays to determine a valid free point in the grid.
        - With probability `p`, the function returns the goal point instead of a random valid point.
        - This method assumes that `valid_rows` and `valid_cols` represent the valid free spaces in the grid.
        """

        rand_idx = np.random.choice(len(valid_cols)) # Select random index

        # Select random valid point
        x = valid_rows[rand_idx] 
        y = valid_cols[rand_idx]

        # Select goal with probablity p
        if np.random.uniform(0,1) < p:
            return self.goal
        else:
            return Point(x,y)
        
    def nearest_vertex(self,qrand,tree):
        """
        Finds the nearest vertex in the tree to a randomly sampled point.

        Parameters:
        - qrand (Point): The randomly sampled point for which the nearest vertex in the tree is to be found.
        - tree (dict): A dictionary representing the RRT tree, where keys are vertices (Points) and values are their parent vertices.

        Returns:
        - Point: The nearest vertex in the tree to the given random point `qrand`.

        Notes:
        - The function calculates the Euclidean distance between the random point and all vertices in the tree, selecting the vertex with the minimum distance.
        """

        min_distance = np.inf
        # Loop over all existing Points and find nearest to qrand
        for point in tree.keys():
            if qrand.dist(point) < min_distance and qrand.dist(point)!=0:
                min_distance = qrand.dist(point)
                qnearest = point

        return qnearest
    
    def get_qnew(self,qrand,qnear,dq):

        """
        Steers from a nearest vertex toward a random point, generating a new vertex within a step size.

        Parameters:
        - qrand (Point): The randomly sampled point toward which the new point is steered.
        - qnear (Point): The nearest vertex in the tree to `qrand`.
        - dq (float): The maximum allowable step size for the new vertex.

        Returns:
        - Point: A new vertex (`qnew`) generated by moving from `qnear` in the direction of `qrand` by a distance that is the smaller of `dq` or the actual distance to `qrand`.

        Notes:
        - If `qrand` and `qnear` are the same, `qnear` is returned to avoid division by zero.
        - The function calculates a unit vector in the direction of `qrand` from `qnear` and uses it to determine the new vertex by scaling the step size.
        """

        direction = qnear.vector(qrand) # Steer in direction of qrand from qnear
        distance = direction.norm() # Calculate distance between two points
        unit_vector = direction.unit()
        
        # Check if two points are the same to avoid dividing by zero later
        if distance == 0:
            return qnear

        step = unit_vector*min(dq,distance) # Move in direction of rand with smaller distance between dq and distance
        return qnear.__add__(step)
    
    def is_segment_free(self,p1,p2):

        """
        Checks whether the straight line segment between two points is free of obstacles.

        Parameters:
        - p1 (Point): The starting point of the line segment.
        - p2 (Point): The ending point of the line segment.

        Returns:
        - bool: `True` if the segment is free of obstacles, `False` otherwise.

        Notes:
        - The function divides the line segment into 20 equally spaced points.
        - Each point on the segment is checked against the grid map to ensure it lies in a free space (`gridmap[x, y] == 0`).
        - If any point on the segment lies in an obstacle (`gridmap[x, y] == 1`), the function returns `False`.
        - The coordinates of the points are converted to integers to match the grid map indices.
        """

        p1 = p1.numpy()
        p2 = p2.numpy()

        ps = np.int_(np.linspace(p1,p2,50)) # Divide the line into 20 points
        # Check all points on the line if they are invalid
        for x, y in ps:
            if self.gridmap[x, y] == 1:
                return False

        return True
    
    def reconstruct_path(self,tree,q):

        """
        Reconstructs the path from the goal to the start node using the RRT tree and calculates the total path cost.

        Parameters:
        - tree (dict): A dictionary representing the RRT tree, where keys are child vertices (Points) and values are their parent vertices.
        - q (Point): The goal node from which the path reconstruction begins.

        Returns:
        - tuple: A tuple containing:
            - path (list): A list of vertices (Points) representing the path from the start to the goal.
            - path_cost (float): The total cost of the reconstructed path, calculated as the sum of distances between consecutive points.

        Notes:
        - The function traces the tree from the goal node back to the start by following parent nodes.
        - The resulting path is reversed to order it from start to goal.
        - Path cost is calculated as the sum of Euclidean distances between consecutive vertices in the path.
        """

        current = q # Set current node as end Goal
        path = [current] # Start path from goal
        path_cost = 0  # Initialize path cost to zero

        # Traverse the tree from the Goal node to the Start
        while current in tree.keys():
            current = tree[current]
            path.append(current)
        path.reverse() # Reverse path

        # Add cost of each edge to the path
        for i in range(2,len(path[1:])):
            path_cost += path[i-1].dist(path[i])

        return path[1:],path_cost

    

    def run(self):
        
        """
        Builds the Rapidly-Exploring Random Tree Star (RRT*) to find an optimized path from the start node to the goal node.

        Returns:
        - tuple: A tuple containing:
            - first_tree (dict): The RRT* tree at the iteration when the goal was first reached, where keys are vertices (Points) and values are their parent vertices.
            - first_path (list): A list of vertices (Points) representing the initial path from the start to the goal (excluding the goal node). Returns an empty list if no path is found.
            - first_path_cost (float): The cost of the initial path to the goal. Returns `np.inf` if no path is found.
            - first_iter (int): The iteration number at which the goal was first reached. Returns 0 if no path is found.
            - final_tree (dict): The RRT* tree after all iterations.
            - final_path (list): A list of vertices (Points) representing the final optimized path from the start to the goal. Returns an empty list if no path is found.
            - final_path_cost (float): The cost of the final optimized path to the goal. Returns `np.inf` if no path is found.

        Procedure:
        1. Initialize the tree with the start node as the root (no parent) and set the cost of the start node to 0.
        2. Iteratively:
            - Sample a random point (`qrand`) from the grid using `sample_random_point`.
            - Find the nearest vertex (`qnear`) to `qrand` in the current tree using `nearest_vertex`.
            - Steer toward `qrand` from `qnear` to generate a new point (`qnew`) using `get_qnew`.
            - Check if the segment between `qnear` and `qnew` is obstacle-free using `is_segment_free`.
            - If valid:
                - Add `qnew` to the tree, assigning its parent to minimize cost, and update its cost.
                - Rewire the tree by checking if the cost of neighboring nodes can be reduced through `qnew`.
            - If `qnew` coincides with the goal and the goal is reached for the first time:
                - Save the current tree and path, and record the cost and iteration number.
        3. After all iterations:
            - If the goal is in the tree, reconstruct the optimized path and return it with its cost.
            - Otherwise, return an empty path and infinite cost.

        Notes:
        - The maximum number of iterations is defined by `self.max_iter`.
        - The step size for tree expansion is controlled by `self.dq`.
        - Rewiring optimizes the cost by reassigning parent nodes within a specified distance (`self.max_search_distance`).
        - The path reconstruction is performed using `reconstruct_path`.
        - A path is considered found when `qnew` coincides with the goal node.
        """


        # Initialize the tree with the starting node that has no parent
        tree = {}
        tree[self.start] = None
        first_tree = tree # Copy Tree into first_tree (used to store the first path found)

        # Cost of reaching the start node (this is typically 0)
        costs = {}
        costs[self.start] = 0

        reached_goal = False # Flag to mark when goal is reached for first time

        for i in range(self.max_iter):
            qrand = self.sample_random_point(self.valid_rows,self.valid_cols,self.p) # Sample a point from the grid
            qnear = self.nearest_vertex(qrand,tree) # Find the nearest node to qrand
            qnew = self.get_qnew(qrand,qnear,self.dq) # Create new node in the direction of qrand

            if self.is_segment_free(qnear,qnew): # Check if line between qnear and qnew pass through an object or not
                costs[qnew] = costs[qnear] + qnew.dist(qnear) # update cost of qnew
                qnew_neighbors = [] # Create list of neighbors of qnew
                qmin = qnear

                #cost Optimization
                for j in tree.keys():
                    if qnew.dist(j) < self.max_search_distance and self.is_segment_free(qnew,j):
                        qnew_neighbors.append(j)
                        new_cost = costs[j] + qnew.dist(j) # Calculate cost of qnew from the neighbor j
                        if new_cost < costs[qnew]: # Update cost of qnew if it is less than previous cost
                            qmin = j
                            costs[qnew] = new_cost
                            
                if qnew not in tree:
                    tree[qnew] = qmin # update parent of qnew

                # If goal is reached for the first time, return the tree and path to the goal at that iteration
                if qnew.dist(self.goal)==0 and reached_goal == False:
                    reached_goal = True
                    tree[self.goal] = qnew # add goal to the tree
                    first_tree = copy.deepcopy(tree) # Save the current tree
                    first_path,first_path_cost = self.reconstruct_path(tree,self.goal) # Get the first path reached and its cost
                    costs[self.goal] = first_path_cost 
                    first_iter = i
                    print(f"First path found after {i} iterations")

                # Rewiring
                for neighbor in qnew_neighbors:
                    if neighbor != qmin:
                        new_neighbor_cost = costs[qnew] + qnew.dist(neighbor) 
                        if new_neighbor_cost < costs[neighbor]: # Check if cost of neighbor from qnew is smaller than its previous cost
                            tree[neighbor] = qnew # Update parent of neighbor to qnew
                            costs[neighbor] = new_neighbor_cost #Update cost of neighbor

        if self.goal in tree.keys(): # Check if goal was reached during all iterations
            path,path_cost = self.reconstruct_path(tree,self.goal) # Get path after all iterations and the cost of the final path
            return first_tree, first_path[:-1],first_path_cost,first_iter,tree,path,path_cost
        
        print("No Path Found")
        return {},[],np.inf,0,tree,[],np.inf

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='RRT* Path Planning Algorithm')
    parser.add_argument('map', help='Path to grid map image')
    parser.add_argument('K', type=int, help='Maximum iterations')
    parser.add_argument('dq', type=float, help='Step size')
    parser.add_argument('p', type=float, help='Goal bias probability')
    parser.add_argument('max_search_radius', type=float, help='Maximum search radius for rewiring')
    parser.add_argument('qstart_x', type=float, help='Start x coordinate')
    parser.add_argument('qstart_y', type=float, help='Start y coordinate')
    parser.add_argument('qgoal_x', type=float, help='Goal x coordinate')
    parser.add_argument('qgoal_y', type=float, help='Goal y coordinate')
    
    # Add recording arguments
    parser = add_recording_args(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get command line arguments
    map = args.map
    max_iter = args.K
    dq = args.dq
    p = args.p
    max_search_distance = args.max_search_radius
    start_x = args.qstart_x
    start_y = args.qstart_y
    goal_x = args.qgoal_x
    goal_y = args.qgoal_y

    # Set start and goal points
    start = (start_x,start_y)
    goal = (goal_x,goal_y)

    image = Image.open(map).convert("L")
    gridmap = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    gridmap[gridmap > 0.5] = 1
    gridmap[gridmap <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    gridmap = (gridmap * -1) + 1
    # Print Gridmap without any nodes
    fig = plot(gridmap,start,goal,{},[])
    plt.title(f"{map.split('/')[1].split('.')[0]}",fontsize = 18)
    
    # Setup recording if requested
    recorder = setup_recording(args, fig)
    
    plt.show()

    rrt_star = RRTStar(gridmap,max_iter,dq,p,max_search_distance,start,goal)
    first_tree, first_path,first_path_cost,first_iter,tree,final_path,final_path_cost= rrt_star.run()

    if final_path == []:
        plot_with_recording(gridmap,start,goal,tree,[], title="No Path Found", recorder=recorder)
        plt.show()
    else:
        # Plot First Path
        print("First Path Cost: ", first_path_cost)
        print("Path to follow: ")
        print(*first_path,sep='\n')
        print('\n')
        plot_with_recording(gridmap,start,goal,first_tree,first_path, 
                          title=f"First path found after {first_iter} iterations with cost {round(first_path_cost,2)}", 
                          recorder=recorder)
        plt.show()

        # Plot Final Path
        print("Final Path Cost: ", final_path_cost)
        print("Path to follow: ")
        print(*final_path,sep='\n')
        print('\n')
        plot_with_recording(gridmap,start,goal,tree,final_path, 
                          title=f"Path found after {max_iter} iterations with cost {round(final_path_cost,2)}", 
                          recorder=recorder)
        plt.show()
    
    # Stop recording if active
    if recorder is not None:
        recorder.stop()
