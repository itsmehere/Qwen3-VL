"""
Trajectory Visualization Utilities

This module contains functions for overlaying colored trajectory arrows on images.
"""

import os
import numpy as np
import re
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional


def extract_trajectory_from_text(text: str) -> List[List[int]]:
    """
    Extract trajectory coordinates from model output text.
    
    Args:
        text: The text output from the model containing trajectory information
        
    Returns:
        List of [x, y] coordinate pairs as integers, or empty list if no trajectory found
        
    Examples:
        >>> extract_trajectory_from_text("2D visual trace: [[100, 200], [150, 250]]")
        [[100, 200], [150, 250]]
        >>> extract_trajectory_from_text("Traces: [[100, 200], [150, 250]]")
        [[100, 200], [150, 250]]
        >>> extract_trajectory_from_text("No trajectory here")
        []
    """
    # Look for individual coordinate pairs [x, y] in the text
    # This approach works for both nested [[...]] and individual [...] patterns
    coord_pattern = r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    coord_matches = re.findall(coord_pattern, text)
    
    if not coord_matches:
        return []
    
    trajectory = []
    for x_str, y_str in coord_matches:
        try:
            # Convert to floats then to integers for pixel coordinates
            x = int(float(x_str.strip()))
            y = int(float(y_str.strip()))
            trajectory.append([x, y])
        except (ValueError, IndexError):
            continue
            
    return trajectory


def visualize_trajectory_on_image(
    trajectory: List[List[int]], 
    image_path: str, 
    output_path: str,
    start_color: Tuple[int, int, int] = (0, 255, 0),    # Green (start)
    end_color: Tuple[int, int, int] = (255, 0, 0),      # Red (end)
    line_thickness: int = 4,
    title_suffix: str = "",
    existing_image: Optional[Image.Image] = None
) -> Optional[np.ndarray]:
    """
    Visualize a trajectory as a gradient arrow from start to end on an image.
    
    Args:
        trajectory: List of [x, y] coordinate pairs
        image_path: Path to the input image
        output_path: Path where the output image will be saved
        start_color: RGB color for the start of the trajectory
        end_color: RGB color for the end of the trajectory
        line_thickness: Thickness of the trajectory line
        title_suffix: Optional suffix for the title (not currently used)
        
    Returns:
        numpy array of the output image, or None if trajectory is too short
    """
    
    if not trajectory or len(trajectory) < 2:
        return None
        
    # Load or use existing image
    if existing_image is not None:
        pil_image = existing_image
    else:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        pil_image = image.copy()
    draw = ImageDraw.Draw(pil_image)
    
    # Draw gradient line segments
    num_segments = len(trajectory) - 1
    for i in range(num_segments):
        # Calculate gradient color for this segment
        progress = i / max(1, num_segments - 1)  # 0 to 1
        
        # Interpolate between start_color and end_color
        r = int(start_color[0] * (1 - progress) + end_color[0] * progress)
        g = int(start_color[1] * (1 - progress) + end_color[1] * progress)
        b = int(start_color[2] * (1 - progress) + end_color[2] * progress)
        segment_color = (r, g, b)
        
        start_point = tuple(trajectory[i])
        end_point = tuple(trajectory[i + 1])
        draw.line([start_point, end_point], fill=segment_color, width=line_thickness)
        
    # Draw start marker only
    if len(trajectory) > 0:
        # Start point (colored circle)
        start_x, start_y = trajectory[0]
        start_radius = max(3, line_thickness)
        start_bbox = [start_x - start_radius, start_y - start_radius, 
                     start_x + start_radius, start_y + start_radius]
        draw.ellipse(start_bbox, fill=start_color, outline=(255, 255, 255), width=2)
    
    # Save the image if output path is provided
    if output_path:
        pil_image.save(output_path)
    
    return np.array(pil_image)
