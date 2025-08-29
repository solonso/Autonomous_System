#!/usr/bin/env python3
"""
RRT Recording Module
===================

Provides inline recording functionality for RRT and RRT* algorithms.
Captures matplotlib animations and saves to MP4 using FFMpegWriter.
"""

import os
import time
import argparse
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

class RRTRecorder:
    """Records RRT algorithm animations using Matplotlib's FFMpegWriter."""
    
    def __init__(self, figure, output_path, fps=30, dpi=100):
        self.figure = figure
        self.output_path = os.path.expanduser(output_path)
        self.fps = fps
        self.dpi = dpi
        self._context = None
        self._writer = None
        self.is_recording = False
        
    def start(self):
        """Start recording by entering the FFMpegWriter context."""
        if self.is_recording:
            print("[Recorder] Already recording!")
            return
            
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._writer = FFMpegWriter(fps=self.fps)
        self._context = self._writer.saving(self.figure, self.output_path, dpi=self.dpi)
        self._context.__enter__()
        self.is_recording = True
        print(f"[Recorder] Started: {self.output_path} @ {self.fps} fps")
        
    def capture(self):
        """Capture current frame if recording is active."""
        if self.is_recording and self._writer is not None:
            self._writer.grab_frame()
            
    def stop(self):
        """Stop recording and finalize the video file."""
        if not self.is_recording:
            return
            
        if self._context is not None:
            try:
                self._context.__exit__(None, None, None)
            finally:
                self._context = None
                self._writer = None
                self.is_recording = False
                print(f"[Recorder] Finalized: {self.output_path}")

def add_recording_args(parser):
    """Add recording-related command line arguments to an ArgumentParser."""
    parser.add_argument('--record', action='store_true', 
                       help='Enable inline recording to MP4')
    parser.add_argument('--record-path', type=str, 
                       default=os.path.expanduser('~/Videos/rrt_%Y%m%d_%H%M%S.mp4'),
                       help='Output path for recording (supports strftime tokens)')
    parser.add_argument('--record-fps', type=int, default=30,
                       help='Recording frames per second')
    return parser

def setup_recording(args, figure):
    """Setup recording if --record flag is provided."""
    if not args.record:
        return None
        
    try:
        output_path = time.strftime(args.record_path)
    except Exception:
        output_path = args.record_path
        
    recorder = RRTRecorder(figure=figure, output_path=output_path, fps=args.record_fps)
    recorder.start()
    return recorder

def plot_with_recording(gridmap, start, goal, tree, path, show_vertices=False, 
                       title="RRT Path Planning", recorder=None, delay=0.1):
    """
    Enhanced plot function that captures frames for recording.
    
    Parameters:
    - gridmap: 2D array representing the environment
    - start: (row, col) coordinates of start position
    - goal: (row, col) coordinates of goal position  
    - tree: dictionary representing the tree structure
    - path: list of vertices representing the path
    - show_vertices: if True, displays vertices with indices
    - title: plot title
    - recorder: RRTRecorder instance (optional)
    - delay: pause duration after plotting
    """
    plt.figure(figsize=(10, 10))
    plt.matshow(gridmap, fignum=0)
    
    # Plot vertices
    if show_vertices:
        for i, v in enumerate(tree.keys()):
            plt.plot(v.y, v.x, "+g")
            plt.text(v.y, v.x, i, fontsize=14, color="w")
    
    # Plot edges
    for parent, child in tree.items():
        if child is not None:
            plt.plot([parent.y, child.y], [parent.x, child.x], '-w')
    
    # Plot path
    for i in range(1, len(path)):
        plt.plot([path[i-1].y, path[i].y], [path[i-1].x, path[i].x], 
                "r", linewidth=2)
    
    # Plot start and goal
    plt.plot(start[1], start[0], "r*", markersize=12, label='Start')
    plt.plot(goal[1], goal[0], "b*", markersize=12, label='Goal')
    plt.legend()
    plt.title(title, fontsize=18)
    
    # Capture frame if recording
    if recorder is not None:
        recorder.capture()
    
    plt.pause(delay)
    return plt.gcf()

def plot2_with_recording(gridmap, start, goal, tree, original_path, smooth_path, 
                        show_vertices=False, title="Original vs Smooth Path", 
                        recorder=None, delay=0.1):
    """
    Enhanced plot2 function that captures frames for recording.
    
    Parameters:
    - gridmap: 2D array representing the environment
    - start: (row, col) coordinates of start position
    - goal: (row, col) coordinates of goal position
    - tree: dictionary representing the tree structure
    - original_path: list of vertices for original path
    - smooth_path: list of vertices for smoothed path
    - show_vertices: if True, displays vertices with indices
    - title: plot title
    - recorder: RRTRecorder instance (optional)
    - delay: pause duration after plotting
    """
    plt.figure(figsize=(10, 10))
    plt.matshow(gridmap, fignum=0)
    
    # Plot vertices
    if show_vertices:
        for i, v in enumerate(tree.keys()):
            plt.plot(v.y, v.x, "+g")
            plt.text(v.y, v.x, i, fontsize=14, color="w")
    
    # Plot edges
    for parent, child in tree.items():
        if child is not None:
            plt.plot([parent.y, child.y], [parent.x, child.x], '-w')
    
    # Plot original path
    for i in range(1, len(original_path)):
        plt.plot([original_path[i-1].y, original_path[i].y], 
                [original_path[i-1].x, original_path[i].x], 
                "r", linewidth=2, label='Original Path' if i == 1 else "")
    
    # Plot smooth path
    for i in range(1, len(smooth_path)):
        plt.plot([smooth_path[i-1].y, smooth_path[i].y], 
                [smooth_path[i-1].x, smooth_path[i].x], 
                "y", linewidth=3, label='Smooth Path' if i == 1 else "")
    
    # Plot start and goal
    plt.plot(start[1], start[0], "r*", markersize=12, label='Start')
    plt.plot(goal[1], goal[0], "b*", markersize=12, label='Goal')
    plt.legend()
    plt.title(title, fontsize=18)
    
    # Capture frame if recording
    if recorder is not None:
        recorder.capture()
    
    plt.pause(delay)
    return plt.gcf()
