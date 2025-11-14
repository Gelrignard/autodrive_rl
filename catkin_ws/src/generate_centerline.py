#!/usr/bin/env python3
"""Generate smooth centerline from sparse waypoints."""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


def load_sparse_points(filepath):
    """Load sparse waypoints from text file"""
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    points.append([float(parts[0]), float(parts[1])])
    return np.array(points)


def generate_smooth_centerline(sparse_points, num_points=300):
    """Generate smooth centerline using splines"""
    x = sparse_points[:, 0]
    y = sparse_points[:, 1]
    
    # Close the loop
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    # Fit spline
    tck, u = splprep([x, y], s=0, k=3, per=True)
    
    # Evaluate at num_points
    u_new = np.linspace(0, 1, num_points)
    smooth_x, smooth_y = splev(u_new, tck)
    
    return np.column_stack([smooth_x, smooth_y])


def plot_centerline(sparse_points, smooth_points):
    """Plot sparse and smooth points"""
    plt.figure(figsize=(12, 10))
    
    # Plot smooth centerline
    plt.plot(smooth_points[:, 0], smooth_points[:, 1], 'b-', 
             linewidth=2, label='Smooth Centerline (300 points)')
    
    # Plot sparse points
    plt.plot(sparse_points[:, 0], sparse_points[:, 1], 'ro-', 
             markersize=6, linewidth=1, label='Sparse Points', alpha=0.7)
    
    # Mark start point
    plt.scatter(sparse_points[0, 0], sparse_points[0, 1], 
                s=200, c='green', marker='*', label='Start', zorder=5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Track Centerline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def save_json(points, filepath):
    """Save to JSON"""
    waypoints = [{'x': float(p[0]), 'y': float(p[1])} for p in points]
    data = {
        'waypoints': waypoints,
        'is_closed': True,
        'num_points': len(waypoints)
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(waypoints)} points to {filepath}")


if __name__ == '__main__':
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'sparse_waypoints.txt'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'track_centerline.json'
    
    print(f"Loading {input_file}...")
    sparse = load_sparse_points(input_file)
    print(f"Loaded {len(sparse)} sparse points")
    
    print("Generating 300 smooth points...")
    smooth = generate_smooth_centerline(sparse, num_points=300)
    
    # Plot for visualization
    print("Displaying plot...")
    plot_centerline(sparse, smooth)
    
    # Save to JSON (remove duplicate last point)
    save_json(smooth[:-1], output_file)
    print("Done!")