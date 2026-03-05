from typing import List
import numpy as np

def collate_outputs(outputs: List[dict]):
    """
    Used to consolidate the outputs of default train_step and validation_step.
    This function should be extended if different behavior is required.

    Assumes 'outputs' is a list of dictionaries, where all dictionaries have the same keys.
    For np.ndarray values, they are collected into a list if shapes are inconsistent,
    rather than being merged with np.vstack.
    """
    collated = {}
    # Iterate over the keys of the first output dictionary (assuming all dictionaries share keys).
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            # Get the shapes of all arrays for this key.
            shapes = [o[k].shape for o in outputs]
            # Check if all shapes are identical.
            if len(set(shapes)) == 1:  # Shapes are consistent, can stack.
                collated[k] = np.vstack([o[k][None] for o in outputs])
            else:  # Shapes are inconsistent (e.g., variable patch/batch sizes).
                collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], list):
            # Check if it's a list of strings (for report_texts)
            if len(outputs[0][k]) > 0 and isinstance(outputs[0][k][0], str):
                # For string lists, keep them as nested lists (one per output)
                collated[k] = [o[k] for o in outputs]
            else:
                # For other lists, flatten them
                collated[k] = [item for o in outputs for item in o[k]]
        elif outputs[0][k] is None:
            # Handle None values (skip them or collect as list of Nones)
            collated[k] = [o.get(k) for o in outputs]
        elif isinstance(outputs[0][k], dict):
            # Handle dictionary values (keep them as list of dictionaries)
            collated[k] = [o[k] for o in outputs]
        else:
            raise ValueError(f'Cannot collate inputs of type {type(outputs[0][k])}.'
                             f' Please modify collate_outputs to add this functionality')

    return collated
