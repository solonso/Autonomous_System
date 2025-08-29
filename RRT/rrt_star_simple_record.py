#!/usr/bin/env python3
"""
Simplified RRT* with Recording
==============================

A simplified version of RRT* that records the tree growth and optimization process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from PIL import Image
from Point import Point
import argparse
import os
import time
import copy

def plot_rrt_star_frame(gridmap, start, goal, tree, path=None, iteration=0, max_iter=1000, 
                       title="RRT* Tree Growth"):
    """Plot a single frame of the RRT* tree."""
    plt.clf()
    plt.imshow(gridmap, cmap='gray_r', origin='upper', alpha=0.8)
    
    # Plot tree edges
    for parent, child in tree.items():
        if child is not None:
            plt.plot([parent.y, child.y], [parent.x, child.x], '-w', linewidth=1, alpha=0.7)
    
    # Plot tree nodes
    for node in tree.keys():
        plt.scatter(node.y, node.x, c='white', s=20, alpha=0.8)
    
    # Plot path if provided
    if path and len(path) > 1:
        for i in range(1, len(path)):
            plt.plot([path[i-1].y, path[i].y], [path[i-1].x, path[i].x], 
                    "r", linewidth=3, zorder=8)
    
    # Plot start and goal
    plt.plot(start[1], start[0], "r*", markersize=15, label='Start', zorder=10)
    plt.plot(goal[1], goal[0], "b*", markersize=15, label='Goal', zorder=10)
    
    plt.title(f'{title} - Iteration {iteration}/{max_iter}', fontsize=14, fontweight='bold')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    plt.legend()

class SimpleRRTStar:
    """Simplified RRT* implementation with recording."""
    
    def __init__(self, gridmap, max_iter, dq, p, max_search_distance, start, goal):
        self.gridmap = gridmap
        self.max_iter = max_iter
        self.dq = dq
        self.p = p
        self.max_search_distance = max_search_distance
        self.start = Point(start[0], start[1])
        self.goal = Point(goal[0], goal[1])
        self.valid_rows, self.valid_cols = np.where(self.gridmap == 0)
    
    def sample_random_point(self, valid_rows, valid_cols, p):
        if np.random.random() < p:
            return self.goal
        else:
            idx = np.random.choice(len(valid_rows))
            return Point(valid_rows[idx], valid_cols[idx])
    
    def nearest_vertex(self, qrand, tree):
        min_distance = np.inf
        qnearest = None
        for point in tree.keys():
            if qrand.dist(point) < min_distance and qrand.dist(point) != 0:
                min_distance = qrand.dist(point)
                qnearest = point
        return qnearest
    
    def get_qnew(self, qrand, qnear, dq):
        direction = qnear.vector(qrand)
        distance = direction.norm()
        if distance == 0:
            return qnear
        unit_vector = direction.unit()
        step = unit_vector * min(dq, distance)
        return qnear.__add__(step)
    
    def is_segment_free(self, p1, p2):
        p1 = p1.numpy()
        p2 = p2.numpy()
        ps = np.int_(np.linspace(p1, p2, 20))
        for x, y in ps:
            if self.gridmap[x, y] == 1:
                return False
        return True
    
    def reconstruct_path(self, tree, costs, q):
        current = q
        path = [current]
        path_cost = 0
        
        while current in tree.keys():
            parent = tree[current]
            if parent is not None:
                path_cost += current.dist(parent)
            current = parent
            if current is not None:
                path.append(current)
        path.reverse()
        return path, path_cost
    
    def run_with_recording(self, output_path, fps=10):
        """Run RRT* with recording."""
        tree = {}
        costs = {}
        tree[self.start] = None
        costs[self.start] = 0
        
        # Setup recording
        fig, ax = plt.subplots(figsize=(12, 10))
        writer = FFMpegWriter(fps=fps)
        
        first_tree = None
        first_path = None
        first_path_cost = np.inf
        first_iter = 0
        reached_goal = False
        
        with writer.saving(fig, output_path, dpi=100):
            # Initial frame
            plot_rrt_star_frame(self.gridmap, (self.start.x, self.start.y), 
                              (self.goal.x, self.goal.y), tree, iteration=0, max_iter=self.max_iter)
            writer.grab_frame()
            
            for i in range(self.max_iter):
                qrand = self.sample_random_point(self.valid_rows, self.valid_cols, self.p)
                qnear = self.nearest_vertex(qrand, tree)
                qnew = self.get_qnew(qrand, qnear, self.dq)
                
                if self.is_segment_free(qnear, qnew):
                    # Find minimum cost parent
                    qmin = qnear
                    costs[qnew] = costs[qnear] + qnew.dist(qnear)
                    
                    # Find neighbors and optimize
                    qnew_neighbors = []
                    for j in tree.keys():
                        if qnew.dist(j) < self.max_search_distance and self.is_segment_free(qnew, j):
                            qnew_neighbors.append(j)
                            if j in costs:
                                new_cost = costs[j] + qnew.dist(j)
                                if new_cost < costs[qnew]:
                                    qmin = j
                                    costs[qnew] = new_cost
                    
                    if qnew not in tree:
                        tree[qnew] = qmin
                    
                    # Check if goal is reached for the first time
                    if qnew.dist(self.goal) == 0 and not reached_goal:
                        reached_goal = True
                        tree[self.goal] = qnew
                        first_tree = copy.deepcopy(tree)
                        first_path, first_path_cost = self.reconstruct_path(tree, costs, self.goal)
                        first_iter = i
                        print(f"First path found after {i+1} iterations")
                    
                    # Rewiring
                    for neighbor in qnew_neighbors:
                        if neighbor != qmin and neighbor in costs:
                            new_neighbor_cost = costs[qnew] + qnew.dist(neighbor)
                            if new_neighbor_cost < costs[neighbor]:
                                tree[neighbor] = qnew
                                costs[neighbor] = new_neighbor_cost
                    
                    # Record frame every few iterations
                    if i % 10 == 0:
                        plot_rrt_star_frame(self.gridmap, (self.start.x, self.start.y), 
                                          (self.goal.x, self.goal.y), tree, iteration=i+1, max_iter=self.max_iter)
                        writer.grab_frame()
            
            # Final result
            if self.goal in tree.keys():
                final_path, final_path_cost = self.reconstruct_path(tree, costs, self.goal)
                print(f"Final path cost: {final_path_cost:.2f}")
                
                # Show final tree with path
                plot_rrt_star_frame(self.gridmap, (self.start.x, self.start.y), 
                                  (self.goal.x, self.goal.y), tree, final_path, 
                                  iteration=self.max_iter, max_iter=self.max_iter,
                                  title="RRT* Final Path")
                writer.grab_frame()
                
                return first_tree, first_path, first_path_cost, first_iter, tree, final_path, final_path_cost
            else:
                print("No path found")
                return {}, [], np.inf, 0, tree, [], np.inf

def main():
    parser = argparse.ArgumentParser(description='Simple RRT* with Recording')
    parser.add_argument('map', help='Path to grid map image')
    parser.add_argument('K', type=int, help='Maximum iterations')
    parser.add_argument('dq', type=float, help='Step size')
    parser.add_argument('p', type=float, help='Goal bias probability')
    parser.add_argument('max_search_radius', type=float, help='Maximum search radius for rewiring')
    parser.add_argument('qstart_x', type=float, help='Start x coordinate')
    parser.add_argument('qstart_y', type=float, help='Start y coordinate')
    parser.add_argument('qgoal_x', type=float, help='Goal x coordinate')
    parser.add_argument('qgoal_y', type=float, help='Goal y coordinate')
    parser.add_argument('--output', type=str, default='~/Videos/rrt_star_simple_%Y%m%d_%H%M%S.mp4',
                       help='Output video path')
    parser.add_argument('--fps', type=int, default=10, help='Video FPS')
    
    args = parser.parse_args()
    
    # Load map
    image = Image.open(args.map).convert("L")
    gridmap = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    gridmap[gridmap > 0.5] = 1
    gridmap[gridmap <= 0.5] = 0
    gridmap = (gridmap * -1) + 1
    
    # Setup start and goal
    start = (args.qstart_x, args.qstart_y)
    goal = (args.qgoal_x, args.qgoal_y)
    
    # Expand output path
    output_path = os.path.expanduser(time.strftime(args.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Starting RRT* with recording to: {output_path}")
    
    # Run RRT* with recording
    rrt_star = SimpleRRTStar(gridmap, args.K, args.dq, args.p, args.max_search_radius, start, goal)
    first_tree, first_path, first_cost, first_iter, final_tree, final_path, final_cost = rrt_star.run_with_recording(output_path, args.fps)
    
    print(f"Recording completed: {output_path}")
    print(f"First path found at iteration: {first_iter}")
    print(f"First path cost: {first_cost:.2f}")
    print(f"Final path cost: {final_cost:.2f}")

if __name__ == "__main__":
    main()
