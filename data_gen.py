"""
Synthetic ARC-like data generator for the "translate all objects" primitive.

Generates tasks where:
- Grid contains 1-4 objects (polyomino blobs) + optional single-cell flags + optional thin lines
- All objects translate by the same (dx, dy) vector
- Output is the translated scene

Task format follows ARC JSON style:
{
    "train": [{"input": [[...]], "output": [[...]]}, ...],
    "test": [{"input": [[...]], "output": [[...]]}]
}
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import random


@dataclass
class DataConfig:
    """Configuration for synthetic ARC data generation."""
    grid_size: int = 16
    min_objects: int = 1
    max_objects: int = 4
    min_object_size: int = 3  # cells per object
    max_object_size: int = 8
    n_colors: int = 10  # 0-9, where 0 is background
    max_translation: int = 3  # dx, dy in [-max, max] excluding (0,0)
    min_gap: int = 1  # minimum gap between objects
    
    # Fine detail cases
    flag_prob: float = 0.3  # probability of including single-cell markers
    max_flags: int = 3
    line_prob: float = 0.2  # probability of including thin line object
    
    # Task structure
    n_train_pairs: int = 2
    n_test_pairs: int = 1
    
    seed: Optional[int] = None


def generate_polyomino(size: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """
    Generate a random connected polyomino (4-connected blob) of given size.
    Returns list of (row, col) offsets relative to (0, 0).
    """
    if size <= 0:
        return []
    
    cells = [(0, 0)]
    
    while len(cells) < size:
        # Pick a random existing cell
        base = cells[rng.integers(len(cells))]
        
        # Try to add a neighbor
        neighbors = [
            (base[0] - 1, base[1]),  # up
            (base[0] + 1, base[1]),  # down
            (base[0], base[1] - 1),  # left
            (base[0], base[1] + 1),  # right
        ]
        rng.shuffle(neighbors)
        
        for n in neighbors:
            if n not in cells:
                cells.append(n)
                break
    
    return cells


def normalize_polyomino(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Normalize polyomino so min row/col is 0."""
    if not cells:
        return cells
    min_r = min(c[0] for c in cells)
    min_c = min(c[1] for c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def get_bounding_box(cells: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Get (min_row, min_col, max_row, max_col) of cells."""
    if not cells:
        return (0, 0, 0, 0)
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    return (min(rows), min(cols), max(rows), max(cols))


def can_place_object(
    grid: np.ndarray, 
    cells: List[Tuple[int, int]], 
    top_left: Tuple[int, int],
    min_gap: int = 1
) -> bool:
    """Check if object can be placed without overlap or violating gap constraint."""
    H, W = grid.shape
    r0, c0 = top_left
    
    for dr, dc in cells:
        r, c = r0 + dr, c0 + dc
        
        # Check bounds
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        
        # Check overlap
        if grid[r, c] != 0:
            return False
        
        # Check gap (look at neighbors within min_gap distance)
        for gr in range(max(0, r - min_gap), min(H, r + min_gap + 1)):
            for gc in range(max(0, c - min_gap), min(W, c + min_gap + 1)):
                if (gr, gc) != (r, c) and grid[gr, gc] != 0:
                    # Check if this neighbor is part of the same object being placed
                    if (gr - r0, gc - c0) not in cells:
                        return False
    
    return True


def place_object(
    grid: np.ndarray,
    cells: List[Tuple[int, int]],
    top_left: Tuple[int, int],
    color: int
) -> None:
    """Place object on grid (modifies in place)."""
    r0, c0 = top_left
    for dr, dc in cells:
        grid[r0 + dr, c0 + dc] = color


def generate_line_object(
    length: int, 
    horizontal: bool,
    rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """Generate a thin 1-cell-wide line."""
    if horizontal:
        return [(0, i) for i in range(length)]
    else:
        return [(i, 0) for i in range(length)]


def generate_scene(
    config: DataConfig,
    rng: np.random.Generator,
    grid_size: Optional[int] = None,
    n_objects: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate a scene with multiple objects.
    
    Returns:
        grid: (H, W) array with colors 0-9
        objects: list of dicts with keys 'cells', 'color', 'top_left'
    """
    H = W = grid_size or config.grid_size
    grid = np.zeros((H, W), dtype=np.int32)
    objects = []
    
    # Determine number of objects
    if n_objects is None:
        n_objects = rng.integers(config.min_objects, config.max_objects + 1)
    
    # Available colors (1-9, excluding background 0)
    available_colors = list(range(1, config.n_colors))
    rng.shuffle(available_colors)
    
    # Generate main polyomino objects
    for obj_idx in range(n_objects):
        if not available_colors:
            break
            
        size = rng.integers(config.min_object_size, config.max_object_size + 1)
        cells = normalize_polyomino(generate_polyomino(size, rng))
        color = available_colors.pop()
        
        # Try to place object
        bbox = get_bounding_box(cells)
        obj_h = bbox[2] - bbox[0] + 1
        obj_w = bbox[3] - bbox[1] + 1
        
        placed = False
        for _ in range(100):  # max attempts
            r = rng.integers(0, max(1, H - obj_h))
            c = rng.integers(0, max(1, W - obj_w))
            
            if can_place_object(grid, cells, (r, c), config.min_gap):
                place_object(grid, cells, (r, c), color)
                objects.append({
                    'cells': cells,
                    'color': color,
                    'top_left': (r, c),
                    'type': 'polyomino'
                })
                placed = True
                break
        
        if not placed:
            # Couldn't place, try smaller object
            pass
    
    # Add single-cell flags with some probability
    if rng.random() < config.flag_prob and available_colors:
        n_flags = rng.integers(1, min(config.max_flags + 1, len(available_colors) + 1))
        for _ in range(n_flags):
            if not available_colors:
                break
            color = available_colors.pop()
            cells = [(0, 0)]  # single cell
            
            for _ in range(50):
                r = rng.integers(0, H)
                c = rng.integers(0, W)
                if can_place_object(grid, cells, (r, c), config.min_gap):
                    place_object(grid, cells, (r, c), color)
                    objects.append({
                        'cells': cells,
                        'color': color,
                        'top_left': (r, c),
                        'type': 'flag'
                    })
                    break
    
    # Add thin line with some probability
    if rng.random() < config.line_prob and available_colors:
        color = available_colors.pop()
        horizontal = rng.random() < 0.5
        length = rng.integers(3, 7)
        cells = generate_line_object(length, horizontal, rng)
        
        bbox = get_bounding_box(cells)
        obj_h = bbox[2] - bbox[0] + 1
        obj_w = bbox[3] - bbox[1] + 1
        
        for _ in range(50):
            r = rng.integers(0, max(1, H - obj_h))
            c = rng.integers(0, max(1, W - obj_w))
            if can_place_object(grid, cells, (r, c), config.min_gap):
                place_object(grid, cells, (r, c), color)
                objects.append({
                    'cells': cells,
                    'color': color,
                    'top_left': (r, c),
                    'type': 'line'
                })
                break
    
    return grid, objects


def translate_scene(
    grid: np.ndarray,
    objects: List[Dict],
    dx: int,
    dy: int
) -> np.ndarray:
    """
    Translate all objects by (dx, dy).
    dx = row offset, dy = col offset.
    
    Returns new grid with translated objects.
    Collision rule: later objects overwrite earlier ones.
    """
    H, W = grid.shape
    new_grid = np.zeros_like(grid)
    
    for obj in objects:
        r0, c0 = obj['top_left']
        color = obj['color']
        
        for dr, dc in obj['cells']:
            new_r = r0 + dr + dx
            new_c = c0 + dc + dy
            
            # Only place if in bounds
            if 0 <= new_r < H and 0 <= new_c < W:
                new_grid[new_r, new_c] = color
    
    return new_grid


def sample_valid_translation(
    grid: np.ndarray,
    objects: List[Dict],
    max_t: int,
    rng: np.random.Generator
) -> Optional[Tuple[int, int]]:
    """
    Sample a translation (dx, dy) that keeps all objects in bounds.
    Returns None if no valid translation found.
    """
    H, W = grid.shape
    
    # Find all object cells
    all_cells = []
    for obj in objects:
        r0, c0 = obj['top_left']
        for dr, dc in obj['cells']:
            all_cells.append((r0 + dr, c0 + dc))
    
    if not all_cells:
        return None
    
    # Compute valid translation range
    min_r = min(c[0] for c in all_cells)
    max_r = max(c[0] for c in all_cells)
    min_c = min(c[1] for c in all_cells)
    max_c = max(c[1] for c in all_cells)
    
    # Valid dx range: need min_r + dx >= 0 and max_r + dx < H
    # So -min_r <= dx <= H - 1 - max_r
    dx_min = max(-max_t, -min_r)
    dx_max = min(max_t, H - 1 - max_r)
    
    dy_min = max(-max_t, -min_c)
    dy_max = min(max_t, W - 1 - max_c)
    
    # Sample from valid range, excluding (0, 0)
    valid_translations = []
    for dx in range(dx_min, dx_max + 1):
        for dy in range(dy_min, dy_max + 1):
            if dx != 0 or dy != 0:
                valid_translations.append((dx, dy))
    
    if not valid_translations:
        return None
    
    return valid_translations[rng.integers(len(valid_translations))]


def generate_task(
    config: DataConfig,
    rng: np.random.Generator,
    grid_size: Optional[int] = None,
    n_objects: Optional[int] = None,
) -> Optional[Dict]:
    """
    Generate a single ARC-style task.
    
    Returns dict with 'train' and 'test' lists, each containing
    {'input': [[...]], 'output': [[...]]} dicts.
    
    All examples in a task share the same (dx, dy) translation.
    """
    # Sample translation vector for this task
    for _ in range(20):  # Try multiple times to get a valid task
        # Generate base scene
        input_grid, objects = generate_scene(config, rng, grid_size, n_objects)
        
        if not objects:
            continue
        
        # Sample valid translation
        translation = sample_valid_translation(
            input_grid, objects, config.max_translation, rng
        )
        
        if translation is None:
            continue
        
        dx, dy = translation
        
        # Generate train pairs (same translation, different scenes)
        train_pairs = []
        for _ in range(config.n_train_pairs):
            scene_grid, scene_objects = generate_scene(config, rng, grid_size, n_objects)
            if not scene_objects:
                continue
            
            # Check if translation is valid for this scene
            test_trans = sample_valid_translation(scene_grid, scene_objects, config.max_translation, rng)
            if test_trans is None:
                continue
            
            # Use the task's (dx, dy), but verify it's valid
            all_valid = True
            for obj in scene_objects:
                r0, c0 = obj['top_left']
                for dr, dc in obj['cells']:
                    new_r, new_c = r0 + dr + dx, c0 + dc + dy
                    if not (0 <= new_r < input_grid.shape[0] and 0 <= new_c < input_grid.shape[1]):
                        all_valid = False
                        break
                if not all_valid:
                    break
            
            if not all_valid:
                continue
            
            output_grid = translate_scene(scene_grid, scene_objects, dx, dy)
            train_pairs.append({
                'input': scene_grid.tolist(),
                'output': output_grid.tolist()
            })
        
        if len(train_pairs) < config.n_train_pairs:
            continue
        
        # Generate test pairs
        test_pairs = []
        for _ in range(config.n_test_pairs):
            scene_grid, scene_objects = generate_scene(config, rng, grid_size, n_objects)
            if not scene_objects:
                continue
            
            # Verify translation validity
            all_valid = True
            for obj in scene_objects:
                r0, c0 = obj['top_left']
                for dr, dc in obj['cells']:
                    new_r, new_c = r0 + dr + dx, c0 + dc + dy
                    if not (0 <= new_r < input_grid.shape[0] and 0 <= new_c < input_grid.shape[1]):
                        all_valid = False
                        break
                if not all_valid:
                    break
            
            if not all_valid:
                continue
            
            output_grid = translate_scene(scene_grid, scene_objects, dx, dy)
            test_pairs.append({
                'input': scene_grid.tolist(),
                'output': output_grid.tolist()
            })
        
        if len(test_pairs) < config.n_test_pairs:
            continue
        
        return {
            'train': train_pairs,
            'test': test_pairs,
            'metadata': {
                'translation': [dx, dy],
                'grid_size': grid_size or config.grid_size,
            }
        }
    
    return None


def generate_dataset(
    n_tasks: int,
    config: DataConfig,
    seed: int = 42,
    grid_size: Optional[int] = None,
    n_objects_range: Optional[Tuple[int, int]] = None,
) -> List[Dict]:
    """Generate a dataset of tasks."""
    rng = np.random.default_rng(seed)
    tasks = []
    
    attempts = 0
    while len(tasks) < n_tasks and attempts < n_tasks * 10:
        attempts += 1
        
        # Sample n_objects if range provided
        n_objects = None
        if n_objects_range:
            n_objects = rng.integers(n_objects_range[0], n_objects_range[1] + 1)
        
        task = generate_task(config, rng, grid_size, n_objects)
        if task:
            tasks.append(task)
    
    return tasks


def save_dataset(tasks: List[Dict], path: str):
    """Save dataset to JSON file."""
    with open(path, 'w') as f:
        json.dump(tasks, f, indent=2)


def load_dataset(path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================================
# Batch generation for training (many-shot)
# ============================================================================

def generate_batch_many_shot(
    batch_size: int,
    config: DataConfig,
    rng: np.random.Generator,
    grid_size: Optional[int] = None,
    n_objects_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of (input, output) pairs for many-shot training.
    
    Unlike few-shot ARC tasks, each sample is independent.
    
    Returns:
        inputs: (batch_size, H, W) int array
        outputs: (batch_size, H, W) int array
    """
    H = W = grid_size or config.grid_size
    inputs = np.zeros((batch_size, H, W), dtype=np.int32)
    outputs = np.zeros((batch_size, H, W), dtype=np.int32)
    
    for i in range(batch_size):
        # Sample n_objects
        if n_objects_range:
            n_objects = rng.integers(n_objects_range[0], n_objects_range[1] + 1)
        else:
            n_objects = rng.integers(config.min_objects, config.max_objects + 1)
        
        # Generate scene
        for _ in range(10):  # retry if needed
            input_grid, objects = generate_scene(config, rng, grid_size, n_objects)
            if not objects:
                continue
            
            # Sample translation
            translation = sample_valid_translation(
                input_grid, objects, config.max_translation, rng
            )
            if translation is None:
                continue
            
            dx, dy = translation
            output_grid = translate_scene(input_grid, objects, dx, dy)
            
            inputs[i] = input_grid
            outputs[i] = output_grid
            break
    
    return inputs, outputs


# ============================================================================
# Test / Demo
# ============================================================================

def visualize_grid(grid: np.ndarray) -> str:
    """Simple text visualization of grid."""
    symbols = '.123456789'
    lines = []
    for row in grid:
        line = ''.join(symbols[min(v, 9)] for v in row)
        lines.append(line)
    return '\n'.join(lines)


if __name__ == "__main__":
    print("Testing ARC data generator...")
    
    config = DataConfig(grid_size=16, min_objects=2, max_objects=4)
    rng = np.random.default_rng(42)
    
    # Generate a single task
    task = generate_task(config, rng)
    if task:
        print(f"\nTask translation: {task['metadata']['translation']}")
        print(f"Train pairs: {len(task['train'])}")
        print(f"Test pairs: {len(task['test'])}")
        
        print("\n--- Train Example 0 ---")
        print("Input:")
        print(visualize_grid(np.array(task['train'][0]['input'])))
        print("\nOutput:")
        print(visualize_grid(np.array(task['train'][0]['output'])))
    
    # Generate batch for many-shot training
    print("\n\nGenerating batch for many-shot training...")
    inputs, outputs = generate_batch_many_shot(8, config, rng)
    print(f"Batch shapes: inputs={inputs.shape}, outputs={outputs.shape}")
    
    print("\n--- Batch Sample 0 ---")
    print("Input:")
    print(visualize_grid(inputs[0]))
    print("\nOutput:")
    print(visualize_grid(outputs[0]))
    
    # Test OOD generation
    print("\n\nTesting OOD generation (24x24, 6 objects)...")
    inputs_ood, outputs_ood = generate_batch_many_shot(
        4, config, rng, grid_size=24, n_objects_range=(5, 7)
    )
    print(f"OOD shapes: inputs={inputs_ood.shape}, outputs={outputs_ood.shape}")
    print("\n--- OOD Sample 0 ---")
    print("Input:")
    print(visualize_grid(inputs_ood[0]))
    print("\nOutput:")
    print(visualize_grid(outputs_ood[0]))
