#!/usr/bin/env python3
"""
Working RRT with Real-time Visualization
========================================

A simple, working RRT implementation that shows tree growth in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from PIL import Image
from Point import Point
import time
import os

def load_map(map_path):
    """Load and process the map."""
    image = Image.open(map_path)
    
    # Check if image is RGB/colorful
    if image.mode in ['RGB', 'RGBA']:
        # For colorful maps, convert to grayscale for processing but keep original for display
        gray_image = image.convert("L")
        gridmap = np.array(gray_image.getdata()).reshape(gray_image.size[0], gray_image.size[1]) / 255
        gridmap[gridmap > 0.5] = 1
        gridmap[gridmap <= 0.5] = 0
        gridmap = (gridmap * -1) + 1
        return gridmap, image
    else:
        # For grayscale maps
        gridmap = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
        gridmap[gridmap > 0.5] = 1
        gridmap[gridmap <= 0.5] = 0
        gridmap = (gridmap * -1) + 1
        return gridmap, image

class WorkingRRT:
    def __init__(self, gridmap, original_image, start, goal, max_iter=100, step_size=2.0, goal_bias=0.1, record=False, output_path=None):
        self.gridmap = gridmap
        self.original_image = original_image
        self.start = Point(start[0], start[1])
        self.goal = Point(goal[0], goal[1])
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.valid_rows, self.valid_cols = np.where(self.gridmap == 0)
        
        # Setup recording if requested
        self.record = record
        self.output_path = output_path
        self.writer = None
        
        # Setup visualization
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
        # Setup recording after figure is created
        if self.record and self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.writer = FFMpegWriter(fps=10)
            self.writer.setup(self.fig, self.output_path, dpi=100)
            print(f"Recording to: {self.output_path}")
        
    def setup_plot(self):
        """Setup the initial plot."""
        self.ax.clear()
        
        # Show colorful map using a custom colormap
        # Create a colorful colormap: obstacles in yellow, free space in purple
        from matplotlib.colors import ListedColormap
        colors = ['purple', 'yellow']  # 0=free space (purple), 1=obstacle (yellow)
        custom_cmap = ListedColormap(colors)
        self.ax.imshow(self.gridmap, cmap=custom_cmap, origin='upper', alpha=0.8)
            
        self.ax.plot(self.start.y, self.start.x, "r*", markersize=15, label='Start', zorder=10)
        self.ax.plot(self.goal.y, self.goal.x, "b*", markersize=15, label='Goal', zorder=10)
        self.ax.set_xlim(-0.5, self.gridmap.shape[1]-0.5)
        self.ax.set_ylim(self.gridmap.shape[0]-0.5, -0.5)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_title('RRT Tree Growth', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Keep track of plotted elements
        self.tree_lines = []
        self.node_points = []
        
    def sample_random_point(self):
        """Sample a random point with goal bias."""
        if np.random.random() < self.goal_bias:
            return self.goal
        else:
            idx = np.random.choice(len(self.valid_rows))
            return Point(self.valid_rows[idx], self.valid_cols[idx])
    
    def nearest_vertex(self, qrand, tree):
        """Find nearest vertex in tree."""
        min_distance = np.inf
        qnearest = None
        for point in tree.keys():
            dist = qrand.dist(point)
            if dist < min_distance and dist != 0:
                min_distance = dist
                qnearest = point
        return qnearest
    
    def get_qnew(self, qrand, qnear):
        """Steer from qnear toward qrand."""
        direction = qnear.vector(qrand)
        distance = direction.norm()
        if distance == 0:
            return qnear
        unit_vector = direction.unit()
        step = unit_vector * min(self.step_size, distance)
        return qnear.__add__(step)
    
    def is_segment_free(self, p1, p2):
        """Check if segment is obstacle-free."""
        p1 = p1.numpy()
        p2 = p2.numpy()
        ps = np.int_(np.linspace(p1, p2, 20))
        for x, y in ps:
            if self.gridmap[x, y] == 1:
                return False
        return True
    
    def update_visualization(self, tree, iteration):
        """Update the visualization with current tree."""
        # Clear previous tree elements
        for line in self.tree_lines:
            line.remove()
        for point in self.node_points:
            point.remove()
        self.tree_lines.clear()
        self.node_points.clear()
        
        # Plot tree edges
        for parent, child in tree.items():
            if child is not None:
                line, = self.ax.plot([parent.y, child.y], [parent.x, child.x], 
                                   '-w', linewidth=1, alpha=0.7)
                self.tree_lines.append(line)
        
        # Plot tree nodes
        for node in tree.keys():
            point = self.ax.scatter(node.y, node.x, c='white', s=20, alpha=0.8, zorder=5)
            self.node_points.append(point)
        
        # Update title
        self.ax.set_title(f'RRT Tree Growth - Iteration {iteration}/{self.max_iter}', 
                         fontsize=14, fontweight='bold')
        
        # Force redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Record frame if recording
        if self.record and self.writer:
            self.writer.grab_frame()
        
        plt.pause(0.05)  # Faster updates to see tree growth
    
    def run(self):
        """Run RRT algorithm with real-time visualization."""
        tree = {}
        tree[self.start] = None
        
        print("Starting RRT with real-time visualization...")
        print("You should see the tree growing in real-time!")
        
        for i in range(self.max_iter):
            # Sample random point
            qrand = self.sample_random_point()
            
            # Find nearest vertex
            qnear = self.nearest_vertex(qrand, tree)
            if qnear is None:
                continue
            
            # Steer toward random point
            qnew = self.get_qnew(qrand, qnear)
            
            # Check if path is free
            if self.is_segment_free(qnear, qnew):
                # Add new node to tree
                tree[qnew] = qnear
                
                # Update visualization
                self.update_visualization(tree, i+1)
                
                # Check if goal is reached
                if qnew.dist(self.goal) == 0:
                    tree[self.goal] = qnew
                    print(f"Goal reached in {i+1} iterations!")
                    
                    # Show final path
                    self.show_final_path(tree)
                    return tree, i+1
        
        print("No path found within max iterations")
        return tree, self.max_iter
    
    def show_final_path(self, tree):
        """Show the final path from start to goal."""
        # Reconstruct path
        path = []
        current = self.goal
        while current is not None:
            path.append(current)
            current = tree.get(current)
        path.reverse()
        
        # Plot path
        for i in range(1, len(path)):
            self.ax.plot([path[i-1].y, path[i].y], [path[i-1].x, path[i].x], 
                        "r", linewidth=3, zorder=8)
        
        self.ax.set_title(f'RRT Path Found - {len(path)} waypoints', 
                         fontsize=14, fontweight='bold')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Record final frame
        if self.record and self.writer:
            self.writer.grab_frame()
        
        # Keep recording for 5 seconds after finding the path
        for _ in range(50):  # 50 frames * 0.1 seconds = 5 seconds
            if self.record and self.writer:
                self.writer.grab_frame()
            plt.pause(0.1)

def main():
    # Load map
    map_path = "data/map1.png"  # Use colorful map
    gridmap, original_image = load_map(map_path)
    
    # Define start and goal (using the same coordinates as in the PDF)
    start = (60, 60)
    goal = (90, 60)
    
    # Create and run RRT with recording
    output_path = os.path.expanduser("~/Videos/rrt_working_%Y%m%d_%H%M%S.mp4")
    output_path = time.strftime(output_path)
    
    rrt = WorkingRRT(gridmap, original_image, start, goal, max_iter=2000, step_size=10.0, goal_bias=0.2, 
                    record=True, output_path=output_path)
    tree, iterations = rrt.run()
    
    # Close recording
    if rrt.writer:
        rrt.writer.finish()
        print(f"Recording saved to: {output_path}")
    
    print(f"RRT completed in {iterations} iterations")
    print("Press any key to close...")
    input()
    plt.close()

if __name__ == "__main__":
    main()
