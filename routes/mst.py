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
                'centroid_relative': (round((centroid[0] - min_y) / (max_y - min_y), 2), round((centroid[1] - min_x) / (max_x - min_x), 2)),
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

def testing(base64_string):
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
    (np.int64(11), np.int64(8), np.int64(56)): 0,
    (np.int64(11), np.int64(4), np.int64(30)): 1,
    (np.int64(11), np.int64(7), np.int64(43)): 1,
    (np.int64(11), np.int64(7), np.int64(50)): 2,
    (np.int64(11), np.int64(7), np.int64(51)): 3,
    (np.int64(11), np.int64(8), np.int64(57)): 3,
    (np.int64(11), np.int64(7), np.int64(48)): 4,
    (np.int64(11), np.int64(9), np.int64(58)): 4,
    (np.int64(11), np.int64(7), np.int64(57)): 5,
    (np.int64(11), np.int64(8), np.int64(61)): 5,
    (np.int64(11), np.int64(7), np.int64(58)): 6,
    (np.int64(11), np.int64(8), np.int64(62)): 6,
    (np.int64(11), np.int64(7), np.int64(36)): 7,
    (np.int64(11), np.int64(8), np.int64(40)): 7,
    (np.int64(11), np.int64(7), np.int64(56)): 8,
    (np.int64(11), np.int64(8), np.int64(70)): 8,
}

class Sol():
    def __init__(self, base64_string):
        self.b64 = base64_string
        self.pixels = extract_pixels_from_base64(base64_string)
        self.connected_areas = extract_connected_areas(self.pixels)
    def parse_digits(self, rgb):
        digits = []
        for area in self.connected_areas:
            if area['color'] != rgb:
                continue
            if area['bbox_size'] in bbox_size_to_digit:
                digit = [area['centroid'][1], bbox_size_to_digit[area['bbox_size']]]
                if digit[1] == 6 and min(area['centroid_relative']) > 0.5:
                    digit[1] = 9
                    
                digits.append(tuple(digit))
        digits.sort()
        if len(digits) == 0:
            logger.info(self.b64)
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
        return compute_mst_weight(self.n, self.adj_list)

# TEST_CASE = 0
@app.route("/mst-calculation", methods = ["POST"])
def mst():
    # global TEST_CASE
    data = request.get_json(silent=True) or {}
    # TEST_CASE += 1
    logger.info(f"---PROCESSING TEST---")

    answer = [Sol(test_case['image']).solve() for test_case in data]

    return jsonify(answer)