import os
import json
import logging
from routes import app
from flask import jsonify, request
import math
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from scipy import ndimage
from collections import defaultdict
logger = logging.getLogger(__name__)

def extract_pixels_from_base64(base64_string):
    try:
        # Remove any data URI prefix (e.g., "data:image/png;base64,")
        if base64_string.startswith("data:image/png;base64,"):
            base64_string = base64_string.split(",")[1]

        # Decode the base64 string to bytes
        img_data = base64.b64decode(base64_string)

        # Convert bytes to a PIL Image
        img = Image.open(BytesIO(img_data))

        # Convert image to RGB if it’s not already (handles RGBA, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert image to a NumPy array for pixel data
        pixel_array = np.array(img)

        # pixel_array is now a 3D NumPy array: [height, width, channels]
        # For RGB, channels = 3 (Red, Green, Blue)
        # Example: pixel_array[0, 0] gives [R, G, B] for the top-left pixel

        return pixel_array

    except Exception as e:
        logger.info(f"Error processing image: {e}")
        return None

def extract_connected_areas(pixels, min_size=5, ignore_color=[255, 255, 255]):
    """
    Extracts connected areas of the same color from an RGB image array, ignoring white pixels and small components.
    
    Args:
    - pixels: NumPy array of shape (height, width, 3) with RGB values.
    - min_size: Minimum size (in pixels) for a component to be considered.
    - ignore_color: RGB list of the color to ignore (default: white).
    
    Returns:
    - A list of dictionaries, each containing:
      - 'color': [R, G, B]
      - 'size': Number of pixels in the component
      - 'centroid': (y, x) approximate center
      - 'bbox': (min_y, min_x, max_y, max_x)
    """
    height, width, _ = pixels.shape
    areas = []
    
    # Find unique colors, excluding the ignore_color
    non_ignore_mask = ~np.all(pixels == ignore_color, axis=2)
    unique_colors = np.unique(pixels[non_ignore_mask].reshape(-1, 3), axis=0)
    
    for color in unique_colors:
        # Create binary mask for this color
        color_mask = np.all(pixels == color, axis=2)
        
        # Label connected components (using 8-connectivity by default)
        labels, num_labels = ndimage.label(color_mask)
        
        for label in range(1, num_labels + 1):
            component_mask = (labels == label)
            size = np.sum(component_mask)
            
            if size < min_size:
                continue  # Skip small noise
            
            # Centroid (center of mass)
            centroid = ndimage.center_of_mass(component_mask)
            
            # Bounding box
            y_indices, x_indices = np.where(component_mask)
            min_y, max_y = y_indices.min(), y_indices.max()
            min_x, max_x = x_indices.min(), x_indices.max()
            
            areas.append({
                'color': color.tolist(),
                'size': size,
                'centroid': (round(centroid[0], 2), round(centroid[1], 2)),
                'centroid_relative': (round(centroid[0] - min_y, 2), round(centroid[1] - min_x, 2)),
                'bbox': (min_y, min_x, max_y, max_x)
            })
    
    return areas

def analyze_digit_areas(pixels):
    # Extract connected areas
    connected_areas = extract_connected_areas(pixels)
    
    # Compute bbox dimensions (width, height) and combine with size
    for area in connected_areas:
        bbox = area['bbox']
        # Calculate width (max_x - min_x) and height (max_y - min_y)
        width = bbox[3] - bbox[1]
        height = bbox[2] - bbox[0]
        # Combine width, height, and size into a tuple
        area['bbox_size'] = (width, height, area['size'])
    
    # Sort by size for consistency
    connected_areas = sorted(connected_areas, key=lambda area: area['size'])
    
    # Extract unique (width, height, size) combinations
    unique_combinations = set(area['bbox_size'] for area in connected_areas)
    unique_combinations = sorted(unique_combinations)  # Sort for readability
    
    # Group areas by their bbox_size for analysis
    grouped_by_combination = defaultdict(list)
    for area in connected_areas:
        grouped_by_combination[area['bbox_size']].append(area)
    
    # Prepare output: unique combinations and their associated areas
    result = {
        'unique_combinations': unique_combinations,
        'grouped_areas': {str(k): v for k, v in grouped_by_combination.items()}
    }
    
    return result, connected_areas

def find_black_regions_and_neighbors(pixels):
    """
    Finds connected black regions and their non-black, non-white neighboring colors in a (600, 600, 3) RGB array.
    
    Args:
        pixels: NumPy array of shape (600, 600, 3) with RGB values.
    
    Returns:
        List of dictionaries, each containing:
        - 'size': Number of black pixels in the region.
        - 'bbox': (min_y, min_x, max_y, max_x) bounding box.
        - 'neighbor_colors': Set of [R, G, B] colors (non-black, non-white) adjacent to the region.
    """
    height, width, _ = pixels.shape
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    
    # Step 1: Identify black regions
    black_mask = np.all(pixels == black, axis=2)
    labels, num_labels = ndimage.label(black_mask, structure=np.ones((3, 3)))  # 8-connectivity
    
    regions = []
    
    for label in range(1, num_labels + 1):
        region_mask = (labels == label)
        size = np.sum(region_mask)
        
        # Skip small regions (optional, adjust min_size as needed)
        min_size = 5
        if size < min_size:
            continue
        
        # Bounding box
        y_indices, x_indices = np.where(region_mask)
        min_y, max_y = y_indices.min(), y_indices.max()
        min_x, max_x = x_indices.min(), x_indices.max()
        
        # Step 2: Find neighboring pixels
        neighbor_colors = set()
        
        # Check 8-connected neighbors for each black pixel in the region
        for y, x in zip(y_indices, x_indices):
            # Define 8-connected neighbor coordinates
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    # Check if neighbor is within bounds
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_color = pixels[ny, nx]
                        # Check if neighbor is non-black and non-white
                        if not np.array_equal(neighbor_color, black) and not np.array_equal(neighbor_color, white):
                            neighbor_colors.add(tuple(neighbor_color))
        
        regions.append({
            # 'size': size,
            # 'bbox': (min_y, min_x, max_y, max_x),
            'centroid': ((min_x + max_x) / 2, (min_y + max_y) / 2),
            'mean': ((np.mean(x_indices)), np.mean(y_indices)),
            'neighbor_colors': neighbor_colors
        })
    
    # Sort regions by size for consistency
    regions = sorted(regions, key=lambda x: x['centroid'])
    
    return regions

def extract_edges_from_regions(regions):
    """
    Processes black regions to find edges based on shared neighbor colors.
    
    Assumes each unique non-black, non-white color connects exactly two regions.
    
    Args:
        regions: List of black region dictionaries from find_black_regions_and_neighbors.
    
    Returns:
        List of edges as (region_index1, region_index2, color), where indices are 0-based
        and sorted (index1 < index2), and color is [R, G, B] list.
    """
    # Collect all unique colors and the regions they connect to
    color_to_regions = defaultdict(list)
    
    for idx, region in enumerate(regions):
        for color in region['neighbor_colors']:
            color_to_regions[color].append(idx)
    
    edges = []
    
    for color, connected_indices in color_to_regions.items():
        # Ensure exactly 2 regions per color
        if len(connected_indices) != 2:
            logger.info(f"Warning: Color {color} connects {len(connected_indices)} regions (expected 2). Skipping.")
            assert False
        
        idx1, idx2 = sorted(connected_indices)
        edges.append((idx1, idx2, list(color)))  # Convert tuple back to list for RGB
    
    # Sort edges by the first index for consistency
    edges = sorted(edges)
    
    return edges

def find(parent, x):
    """Find the root of the set containing x with path compression."""
    if parent[x] != x:
        parent[x] = find(parent, parent[x])  # Path compression
    return parent[x]

def union(parent, rank, x, y):
    """Unite two sets by rank."""
    px, py = find(parent, x), find(parent, y)
    if px == py:
        return
    if rank[px] < rank[py]:
        parent[px] = py
    elif rank[px] > rank[py]:
        parent[py] = px
    else:
        parent[py] = px
        rank[px] += 1

def compute_mst_weight(n, adj_list):
    """
    Compute the total weight of the Minimum Spanning Tree using Kruskal's algorithm.
    
    Args:
        n: Number of vertices (0 to n-1).
        adj_list: List of tuples (u, v, w) representing an edge from u to v with weight w.
                  Assumes undirected graph (each edge appears once).
    
    Returns:
        Total weight of the MST, or -1 if no MST exists (graph is disconnected).
    """
    # Convert adjacency list to edge list (u, v, w)
    edges = []
    for u in range(n):
        for v, w in adj_list[u]:
            if u < v:  # Avoid duplicates in undirected graph
                edges.append((w, u, v))
    
    # Sort edges by weight
    edges.sort()
    
    # Initialize Union-Find data structures
    parent = list(range(n))
    rank = [0] * n
    
    total_weight = 0
    edges_used = 0
    
    # Process edges in sorted order
    for weight, u, v in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            total_weight += weight
            edges_used += 1
    
    # Check if MST exists (must have n-1 edges for n vertices)
    if edges_used != n - 1:
        return -1  # Graph is disconnected or invalid
    
    return total_weight

def testing():
    # graph_0
    # base64_string = "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"
    # graph_3
    base64_string = "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"
    # base64_string = ""

    # Extract pixels
    pixels = extract_pixels_from_base64(base64_string)
    assert pixels is not None

    logger.info(f"Pixel array shape: {pixels.shape}")

    h, w = pixels.shape[:-1]

    result, connected_areas = analyze_digit_areas(pixels)
    connected_areas = sorted(connected_areas, key=lambda area: area['size'])
    for area in connected_areas:
        bbox = area['bbox']
        area['bbox_size'] = (bbox[2] - bbox[0], bbox[3] - bbox[1], area['size'])
        area.pop('size')
        area.pop('bbox')

    # logger.info the extracted areas
    regions = find_black_regions_and_neighbors(pixels)
    edges = extract_edges_from_regions(regions)
    


bbox_size_to_digit = {
    (np.int64(11), np.int64(7), np.int64(56)): 0,
    (np.int64(11), np.int64(4), np.int64(30)): 1,
    (np.int64(11), np.int64(7), np.int64(50)): 2,
    (np.int64(11), np.int64(7), np.int64(51)): 3,
    (np.int64(11), np.int64(7), np.int64(48)): 4,
    (np.int64(11), np.int64(7), np.int64(57)): 5,
    (np.int64(11), np.int64(7), np.int64(58)): 6,
    (np.int64(11), np.int64(7), np.int64(36)): 7,
    (np.int64(11), np.int64(7), np.int64(56)): 8,
}

class Sol():
    def __init__(self, base64_string):
        self.pixels = extract_pixels_from_base64(base64_string)
        self.connected_areas = extract_connected_areas(self.pixels)
    def parse_digits(self, rgb):
        digits = []
        for area in self.connected_areas:
            if area['color'] != rgb:
                continue
            if area['bbox_size'] in bbox_size_to_digit:
                digit = [area['centroid'][1], bbox_size_to_digit[area['bbox_size']]]
                if digit[1] == 6 and area['centroid_relative'] != (np.float64(5.38), np.float64(3.43)):
                    digit[1] = 9
                digits.append(tuple(digit))
        digits.sort()
        assert len(digits) > 0
        edge_weight = 0
        for _, d in digits:
            edge_weight = edge_weight * 10 + d
        
        return edge_weight

    def parse_graph(self):
        for area in self.connected_areas:
            bbox = area['bbox']
            area['bbox_size'] = (bbox[2] - bbox[0], bbox[3] - bbox[1], area['size'])
            area.pop('size')
            area.pop('bbox')
        
        self.regions = find_black_regions_and_neighbors(self.pixels)
        edges = extract_edges_from_regions(self.regions)

        self.n = len(self.regions)
        self.adj_list = [[] for _ in range(self.n)]
        for u, v, rgb in edges:
            w = self.parse_digits(rgb)
            self.adj_list[u].append((v, w))
            self.adj_list[v].append((u, w))
    
    def solve(self):
        self.parse_graph()
        logger.info([(i, region['centroid']) for i, region in enumerate(self.regions)])
        logger.info(self.adj_list)
        return {'value': compute_mst_weight(self.n, self.adj_list)}

# TEST_CASE = 0
@app.route("/mst-calculation", methods = ["POST"])
def mst():
    # global TEST_CASE
    data = request.get_json(silent=True) or {}
    # TEST_CASE += 1
    logger.info(f"---PROCESSING TEST---")

    answer = [Sol(test_case['image']).solve() for test_case in data]

    return jsonify(answer)