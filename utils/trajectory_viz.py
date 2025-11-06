"""
Trajectory Visualization Utilities

This module contains functions for overlaying colored trajectory arrows on images.
"""

import os
import numpy as np
import re
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional


def extract_trajectory_from_text(text: str) -> List[List[List[int]]]:
    """Return a list of trajectories parsed from lines starting with "Traces:".

    Example input:
      "Traces: [[1,2], [3,4]]\nTraces: [[3,4],[5,6]]"
    Returns:
      [[[1,2],[3,4]], [[3,4],[5,6]]]
    """
    traces: List[List[List[int]]] = []
    for line in text.splitlines():
        if line.startswith("Traces:"):
            coords = re.findall(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", line)
            traj = [[int(x), int(y)] for x, y in coords]
            traces.append(traj)
    return traces


def visualize_trajectory_on_image(
    trajectory: List[List[int]], 
    image_path: Optional[str] = None, 
    output_path: Optional[str] = None,
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
        
    # Load or use existing image
    if existing_image is not None:
        if isinstance(existing_image, np.ndarray):
            pil_image = Image.fromarray(existing_image)
        else:
            pil_image = existing_image
    else:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        pil_image = image.copy()
    draw = ImageDraw.Draw(pil_image)

    if not trajectory or len(trajectory) < 2:
        return np.array(pil_image)
    
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
