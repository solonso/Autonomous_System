import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Point import Point
import sys
import argparse
from rrt_recorder import add_recording_args, setup_recording, plot_with_recording, plot2_with_recording

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
            "r",linewidth=2
        )
    # Start
    plt.plot(start[1], start[0], "r*",markersize=12,label='Start')
    # Goal
    plt.plot(goal[1] ,goal[0], "b*",markersize=12,label='Goal')
    plt.legend()
    
    return fig
    
    return fig

def plot2(gridmap,start,goal,tree,original_path,smooth_path,show_vertcies=False):
    """
    Visualizes the grid map, tree structure, original path, and smoothed path, with optional vertex labeling.

    Parameters:
    - gridmap (2D array): The grid map representing the environment, where `0` indicates free spaces and non-zero values indicate obstacles.
    - start (tuple): The (row, col) coordinates of the start position.
    - goal (tuple): The (row, col) coordinates of the goal position.
    - tree (dict): A dictionary where keys are parent vertices and values are child vertices, representing the tree structure.
    - original_path (list): A list of vertices (nodes) representing the original path from start to goal.
    - smooth_path (list): A list of vertices (nodes) representing the smoothed path after optimization.
    - show_vertcies (bool): If True, displays vertices with their indices. Default is False.

    Returns:
    - matplotlib.figure.Figure: The figure object for further manipulation.

    Notes:
    - The grid map is displayed using `matshow`, providing a visual background for the environment.
    - Tree edges are plotted as white lines between parent and child nodes.
    - The original path is displayed as red lines labeled "Original-Path."
    - The smoothed path is displayed as yellow lines labeled "Smooth-Path."
    - If `show_vertcies` is True, each vertex is marked with a green plus sign, and its index is shown next to it.
    - Start and goal positions are marked with distinct symbols (e.g., `r*` for start and `g*` for goal).
    - A legend is automatically added to differentiate the original and smoothed paths.
    """

    fig = plt.figure(figsize=(10, 10))
    plt.matshow(gridmap, fignum=0)

    # Plot Vertcies
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
    # Plot Original Path
    for i in range(1, len(original_path)):
        plt.plot(
            [original_path[i - 1].y, original_path[i].y],
            [original_path[i - 1].x, original_path[i].x],
            "r", linewidth=2,label = "Original-Path" if i==1 else ""
        )
    # Plot Smooth Path
    for i in range(1, len(smooth_path)):
        plt.plot(
            [smooth_path[i - 1].y, smooth_path[i].y],
            [smooth_path[i - 1].x, smooth_path[i].x],
            "g", linewidth=2,label = "Smooth-Path" if i==1 else ""
        )

    # Start
    plt.plot(start[1], start[0], "r*",markersize=12,label='Start')
    # Goal
    plt.plot(goal[1] ,goal[0], "b*",markersize=12,label='Goal')
    plt.legend()

class RRT:

    """
    A class to implement the Rapidly-Exploring Random Tree (RRT) algorithm for path planning in a grid-based environment.

    Attributes:
        gridmap (np.ndarray): 2D numpy array representing the grid map of the environment. 
                              Cells with value 0 are valid configurations; others are obstacles.
        max_iter (int): The maximum number of iterations to grow the RRT.
        dq (float): Step size for extending the tree.
        p (float): Probability of selecting the goal as the target in a given iteration (goal bias).
        start (Point): Starting point for the RRT algorithm, initialized as a Point object.
        goal (Point): Goal point for the RRT algorithm, initialized as a Point object.
        valid_rows (np.ndarray): Array of row indices for valid configurations in the grid map.
        valid_cols (np.ndarray): Array of column indices for valid configurations in the grid map.

    """

    def __init__(self,gridmap,max_iter,dq,p,start,goal):
        self.gridmap = gridmap
        self.max_iter = max_iter
        self.dq = dq
        self.p = p
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
            if qrand.dist(point) < min_distance and qrand.dist(point)!=0: # Ensure nearest vertex is not qrand 
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

        return path[1:-1],path_cost

    def run(self, visualize=False, recorder=None, viz_interval=5):
        """
        Builds the Rapidly-Exploring Random Tree (RRT) to find a path from the start node to the goal node.

        Parameters:
        - visualize (bool): If True, shows real-time tree growth visualization
        - recorder: RRTRecorder instance for capturing frames
        - viz_interval (int): How often to update visualization (every N iterations)

        Returns:
        - tuple: A tuple containing:
            - tree (dict): The constructed RRT tree, where keys are vertices (Points) and values are their parent vertices.
            - path (list): A list of vertices (Points) representing the path from the start to the goal. Returns an empty list if no path is found.
            - path_cost (float): The total cost of the path, calculated as the sum of distances between consecutive points. Returns `np.inf` if no path is found.

        Procedure:
        1. Initialize the tree with the start node as the root (no parent).
        2. Iteratively:
            - Sample a random point (`qrand`) from the grid using `sample_random_point`.
            - Find the nearest vertex (`qnear`) to `qrand` in the current tree using `nearest_vertex`.
            - Steer toward `qrand` from `qnear` to generate a new point (`qnew`) using `get_qnew`.
            - Check if the segment between `qnear` and `qnew` is obstacle-free using `is_segment_free`.
            - If valid, add `qnew` to the tree with `qnear` as its parent.
            - If `qnew` coincides with the goal, reconstruct the path and return the result.
        3. If the maximum number of iterations is reached without finding a path, return the tree, an empty path, and an infinite path cost.

        Notes:
        - The maximum number of iterations is defined by `self.max_iter`.
        - The step size for tree expansion is controlled by `self.dq`.
        - A path is considered found when `qnew` coincides with the goal node.
        - The path reconstruction is performed using `reconstruct_path`.
        """

        # Initialize the tree with the starting node that has no parent
        tree = {}
        tree[self.start] = None
        
        # Initialize real-time visualization if needed
        fig = None
        ax = None
        tree_lines = []
        node_points = []
        title_text = None
        
        if visualize:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(self.gridmap, cmap='gray_r', origin='upper', alpha=0.8)
            ax.plot(self.start.y, self.start.x, "r*", markersize=15, label='Start', zorder=10)
            ax.plot(self.goal.y, self.goal.x, "b*", markersize=15, label='Goal', zorder=10)
            ax.set_xlim(-0.5, self.gridmap.shape[1]-0.5)
            ax.set_ylim(self.gridmap.shape[0]-0.5, -0.5)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.legend()
            title_text = ax.text(0.02, 1.02, '', transform=ax.transAxes,
                               fontsize=14, fontweight='bold', va='bottom')
            plt.tight_layout()

        for i in range(self.max_iter):
            qrand = self.sample_random_point(self.valid_rows,self.valid_cols,self.p) # Sample a point from the grid
            qnear = self.nearest_vertex(qrand,tree) # Find the nearest node to qrand
            qnew = self.get_qnew(qrand,qnear,self.dq) # Create a new node in the direction of qrand
            if self.is_segment_free(qnear,qnew):# Check if line between qnear and qnew doesn't pass through and obstacle
                tree[qnew] = qnear # Add qnew to the tree
                
                # Real-time visualization - show every new node added
                if visualize:
                    # Clear previous tree elements
                    for line in tree_lines:
                        line.remove()
                    for point in node_points:
                        point.remove()
                    tree_lines.clear()
                    node_points.clear()
                    
                    # Plot tree edges
                    for parent, child in tree.items():
                        if child is not None:
                            line, = ax.plot([parent.y, child.y], [parent.x, child.x], 
                                          '-w', linewidth=1, alpha=0.7)
                            tree_lines.append(line)
                    
                    # Plot tree nodes
                    for node in tree.keys():
                        point = ax.scatter(node.y, node.x, c='white', s=20, alpha=0.8, zorder=5)
                        node_points.append(point)
                    
                    # Update title
                    title_text.set_text(f'RRT Tree Growth - Iteration {i+1}/{self.max_iter}')
                    
                    # Force redraw
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
                    if recorder is not None:
                        recorder.capture()
                    
                    plt.pause(0.05)  # Pause to show the update
                
                if qnew.dist(self.goal)==0: # If qnew is the Goal, return the path
                    tree[self.goal] = qnew
                    path,path_cost = self.reconstruct_path(tree,self.goal)
                    print("Path found in " + str(i) + " iterations")
                    
                    # Show final tree with path
                    if visualize:
                        # Clear previous elements
                        for line in tree_lines:
                            line.remove()
                        for point in node_points:
                            point.remove()
                        tree_lines.clear()
                        node_points.clear()
                        
                        # Plot final tree
                        for parent, child in tree.items():
                            if child is not None:
                                line, = ax.plot([parent.y, child.y], [parent.x, child.x], 
                                              '-w', linewidth=1, alpha=0.7)
                                tree_lines.append(line)
                        
                        for node in tree.keys():
                            point = ax.scatter(node.y, node.x, c='white', s=20, alpha=0.8, zorder=5)
                            node_points.append(point)
                        
                        # Plot path
                        for i in range(1, len(path)):
                            ax.plot([path[i-1].y, path[i].y], [path[i-1].x, path[i].x], 
                                   "r", linewidth=3, zorder=8)
                        
                        title_text.set_text(f'RRT Path Found - {i+1} iterations')
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        
                        if recorder is not None:
                            recorder.capture()
                        
                        time.sleep(1)  # Pause to show the final result
                    
                    return tree, path,path_cost,i+1
        
        print("No Path Found")
        
        # Show final tree without path
        if visualize:
            # Clear previous elements
            for line in tree_lines:
                line.remove()
            for point in node_points:
                point.remove()
            tree_lines.clear()
            node_points.clear()
            
            # Plot final tree
            for parent, child in tree.items():
                if child is not None:
                    line, = ax.plot([parent.y, child.y], [parent.x, child.x], 
                                  '-w', linewidth=1, alpha=0.7)
                    tree_lines.append(line)
            
            for node in tree.keys():
                point = ax.scatter(node.y, node.x, c='white', s=20, alpha=0.8, zorder=5)
                node_points.append(point)
            
            title_text.set_text(f'RRT No Path Found - {self.max_iter} iterations')
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if recorder is not None:
                recorder.capture()
            
            time.sleep(1)  # Pause to show the final result
        
        return tree,[],np.inf,i
    
    def smooth(self,path):
        
        """
        Optimizes the given path by removing unnecessary waypoints while ensuring obstacle-free traversal.

        Parameters:
        - path (list): A list of vertices (Points) representing the path to be smoothed. The path must go from the start to the goal.

        Returns:
        - tuple: A tuple containing:
            - smooth_path (list): A list of vertices (Points) representing the smoothed path.
            - smooth_path_cost (float): The total cost of the smoothed path, calculated as the sum of distances between consecutive points.

        Notes:
        - If the input path is empty, the function returns an empty path and an infinite cost.
        - The smoothing process starts at the goal and iteratively checks if a direct path exists to earlier waypoints in the original path.
        - If a direct, obstacle-free path exists between two nodes, intermediate waypoints are removed.
        - The total cost of the smoothed path is calculated as the sum of Euclidean distances between consecutive nodes in the smoothed path.
        - The function ensures the resulting path is valid and free of obstacles by using `is_segment_free`.
        - The final smoothed path is returned in the correct order (from start to goal).
        """

        if path == []: # return empty path is no path was found
            return [],np.inf
        
        next_node = path[-1] #Set as goal
        i = 0
        smooth_path = [path[-1]] # Add goal to smooth-path
        smooth_path_cost = 0

        while smooth_path[-1]!=path[0]: # Check if start is reached
            if self.is_segment_free(path[i],next_node): # Check if a direct path is free from node i to next_node
                smooth_path.append(path[i]) 
                smooth_path_cost += path[i].dist(next_node)
                next_node = path[i] # Set next_node as node i and repeat
                i = 0
            else:
                i+=1
        smooth_path.reverse() # Reverse path 
        return smooth_path,smooth_path_cost

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='RRT Path Planning Algorithm')
    parser.add_argument('map', help='Path to grid map image')
    parser.add_argument('K', type=int, help='Maximum iterations')
    parser.add_argument('dq', type=float, help='Step size')
    parser.add_argument('p', type=float, help='Goal bias probability')
    parser.add_argument('qstart_x', type=float, help='Start x coordinate')
    parser.add_argument('qstart_y', type=float, help='Start y coordinate')
    parser.add_argument('qgoal_x', type=float, help='Goal x coordinate')
    parser.add_argument('qgoal_y', type=float, help='Goal y coordinate')
    
    # Add recording arguments
    parser = add_recording_args(parser)
    parser.add_argument('--visualize', action='store_true', 
                       help='Enable real-time tree growth visualization (without recording)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get command line arguments
    map = args.map
    max_iter = args.K
    dq = args.dq
    p = args.p
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

    rrt = RRT(gridmap,max_iter,dq,p,start,goal)

    # Run RRT with real-time visualization if recording or visualization is enabled
    visualize = args.record or args.visualize
    tree,path,path_cost,i = rrt.run(visualize=visualize, recorder=recorder, viz_interval=5)
    smooth_path,smooth_path_cost = rrt.smooth(path)
    

    if path == []:
        plot_with_recording(gridmap,start,goal,tree,[], title="No Path Found", recorder=recorder)
        plt.show()
    else:
        # Plot Original Path
        print("Total Path Cost: ", path_cost)
        print("Path to follow: ")
        print(*path,sep='\n')
        print('\n')
        plot_with_recording(gridmap,start,goal,tree,path, 
                          title=f"Original Path with cost {round(path_cost,2)} found in {i} iterations", 
                          recorder=recorder)
        plt.show()

        # Plot Smooth Path
        print("Smooth Path Cost: ", smooth_path_cost)
        print("Smooth Path: ")
        print(*smooth_path,sep='\n')
        plot_with_recording(gridmap,start,goal,tree,smooth_path, 
                          title=f"Smooth Path with cost {round(smooth_path_cost,2)}", 
                          recorder=recorder)
        plt.show()

        # Overlay both paths
        plot2_with_recording(gridmap,start,goal,tree,path,smooth_path, 
                           title="Original-Path vs Smooth-Path", 
                           recorder=recorder)
        plt.show()
    
    # Stop recording if active
    if recorder is not None:
        recorder.stop()
        