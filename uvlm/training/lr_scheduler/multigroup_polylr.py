import torch


class MultiGroupPolyLRScheduler:
    """
    Polynomial learning rate scheduler that supports multiple parameter groups.
    Each parameter group decays from its own initial_lr to 0 using polynomial decay.
    """
    def __init__(self, optimizer, max_steps: int, exponent: float = 0.9):
        """
        Args:
            optimizer: PyTorch optimizer with parameter groups
            max_steps: Maximum number of steps for polynomial decay
            exponent: Exponent for polynomial decay (default: 0.9)
        """
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.exponent = exponent
        self.current_step = -1

    def step(self, step=None):
        """
        Update learning rates for all parameter groups using polynomial decay.

        Args:
            step: Current step (if None, increments internal counter)
        """
        if step is None:
            self.current_step += 1
            step = self.current_step
        else:
            self.current_step = step

        for group in self.optimizer.param_groups:
            initial_lr = group.get('initial_lr', group['lr'])
            # Polynomial decay: lr = initial_lr * (1 - step/max_steps)^exponent
            progress = min(step / self.max_steps, 1.0)  # Clamp to [0, 1]
            factor = (1 - progress) ** self.exponent
            group['lr'] = initial_lr * factor

    def get_last_lr(self):
        """Get current learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Get scheduler state."""
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'exponent': self.exponent
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self.max_steps = state_dict.get('max_steps', self.max_steps)
        self.exponent = state_dict.get('exponent', self.exponent)
