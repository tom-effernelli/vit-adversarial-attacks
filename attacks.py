"""Small, explicit L-infinity attacks in pixel space [0, 1]."""

import torch
import torch.nn.functional as F


def _loss(model, pixels, labels, normalize):
    logits = model(pixel_values=normalize(pixels)).logits
    return F.cross_entropy(logits, labels)


def fgsm(model, pixels, labels, normalize, epsilon):
    """One gradient step from the original images."""
    x = pixels.detach().clone().requires_grad_(True)
    gradient = torch.autograd.grad(_loss(model, x, labels, normalize), x)[0]
    return (x + epsilon * gradient.sign()).clamp(0.0, 1.0).detach()


def pgd(model, pixels, labels, normalize, epsilon, step_size, steps, random_start=True):
    """Repeated gradient steps, projected back to each original image's L-inf ball."""
    original = pixels.detach()
    if random_start:
        current = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        current = current.clamp(0.0, 1.0)
    else:
        current = original.clone()

    lower = (original - epsilon).clamp(0.0, 1.0)
    upper = (original + epsilon).clamp(0.0, 1.0)
    for _ in range(steps):
        current = current.detach().requires_grad_(True)
        gradient = torch.autograd.grad(
            _loss(model, current, labels, normalize), current
        )[0]
        proposal = current + step_size * gradient.sign()
        current = torch.maximum(torch.minimum(proposal, upper), lower).detach()
    return current


def random_noise(pixels, epsilon):
    """A same-budget, non-optimized comparison for FGSM and PGD."""
    noise = torch.empty_like(pixels).uniform_(-epsilon, epsilon)
    return (pixels + noise).clamp(0.0, 1.0)

