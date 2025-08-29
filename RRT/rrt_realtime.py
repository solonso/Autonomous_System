#!/usr/bin/env python3
"""
Real-time RRT Visualization
===========================

Provides efficient real-time visualization for RRT tree growth
by updating the same matplotlib figure instead of creating new ones.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

class RealtimeRRTVisualizer:
    """Real-time RRT tree visualization with efficient figure updates."""
    
    def __init__(self, gridmap, start, goal, figsize=(12, 10)):
        self.gridmap = gridmap
        self.start = start
        self.goal = goal
        
        # Setup figure and axes
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.setup_plot()
        
        # Keep track of plotted elements for efficient updates
        self.tree_lines = []
        self.node_points = []
        self.title_text = None
        
    def setup_plot(self):
        """Setup the initial plot with map, start, and goal."""
        self.ax.clear()
        
        # Show the gridmap
        self.ax.imshow(self.gridmap, cmap='gray_r', origin='upper', alpha=0.8)
        
        # Plot start and goal
        self.ax.plot(self.start[1], self.start[0], "r*", markersize=15, label='Start', zorder=10)
        self.ax.plot(self.goal[1], self.goal[0], "b*", markersize=15, label='Goal', zorder=10)
        
        # Add title
        self.title_text = self.ax.text(0.02, 1.02, '', transform=self.ax.transAxes,
                                      fontsize=14, fontweight='bold', va='bottom')
        
        # Setup axes
        self.ax.set_xlim(-0.5, self.gridmap.shape[1]-0.5)
        self.ax.set_ylim(self.gridmap.shape[0]-0.5, -0.5)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        plt.tight_layout()
        
    def update_tree(self, tree, iteration, max_iter, path=None):
        """Update the tree visualization efficiently."""
        # Clear previous tree elements
        for line in self.tree_lines:
            line.remove()
        for point in self.node_points:
            point.remove()
        self.tree_lines.clear()
        self.node_points.clear()
        
        # Ensure title text exists
        if self.title_text is None:
            self.title_text = self.ax.text(0.02, 1.02, '', transform=self.ax.transAxes,
                                          fontsize=14, fontweight='bold', va='bottom')
        
        # Plot tree edges
        for parent, child in tree.items():
            if child is not None:
                line, = self.ax.plot([parent.y, child.y], [parent.x, child.x], 
                                   '-w', linewidth=1, alpha=0.7)
                self.tree_lines.append(line)
        
        # Plot tree nodes (small white dots)
        for node in tree.keys():
            point = self.ax.scatter(node.y, node.x, c='white', s=20, alpha=0.8, zorder=5)
            self.node_points.append(point)
        
        # Plot path if provided
        if path and len(path) > 1:
            for i in range(1, len(path)):
                self.ax.plot([path[i-1].y, path[i].y], [path[i-1].x, path[i].x], 
                           "r", linewidth=3, zorder=8)
        
        # Update title
        if path:
            self.title_text.set_text(f'RRT Path Found - Iteration {iteration}/{max_iter}')
        else:
            self.title_text.set_text(f'RRT Tree Growth - Iteration {iteration}/{max_iter}')
        
        # Force redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        """Close the figure."""
        plt.ioff()
        plt.close(self.fig)

def plot_realtime_rrt(gridmap, start, goal, tree, path=None, iteration=0, max_iter=1000, 
                     title="RRT Tree Growth", delay=0.01):
    """
    Simple real-time RRT plotting function that creates a new figure each time.
    This is less efficient but simpler for debugging.
    """
    plt.figure(figsize=(12, 10))
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
    
    plt.pause(delay)
    return plt.gcf()
