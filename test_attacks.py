"""Checks that the attacks maximize loss while respecting the pixel budget."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from attacks import fgsm, pgd


class TinyClassifier(torch.nn.Module):
    def forward(self, pixel_values):
        average = pixel_values.mean(dim=(1, 2, 3))
        return SimpleNamespace(logits=torch.stack((average, -average), dim=1))


class AttackTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyClassifier()
        self.pixels = torch.full((2, 3, 4, 4), 0.5)
        self.labels = torch.zeros(2, dtype=torch.long)
        self.identity = lambda value: value

    def loss(self, pixels):
        return F.cross_entropy(self.model(pixels).logits, self.labels)

    def assert_valid(self, attacked, epsilon):
        self.assertTrue(torch.all(attacked >= 0))
        self.assertTrue(torch.all(attacked <= 1))
        self.assertLessEqual(float((attacked - self.pixels).abs().max()), epsilon + 1e-6)

    def test_fgsm_increases_loss_and_respects_budget(self):
        attacked = fgsm(self.model, self.pixels, self.labels, self.identity, 0.1)
        self.assert_valid(attacked, 0.1)
        self.assertGreater(float(self.loss(attacked)), float(self.loss(self.pixels)))
        self.assertTrue(torch.allclose(attacked, torch.full_like(attacked, 0.4)))

    def test_pgd_increases_loss_and_projects_every_step(self):
        attacked = pgd(self.model, self.pixels, self.labels, self.identity,
                       epsilon=0.1, step_size=0.07, steps=5, random_start=False)
        self.assert_valid(attacked, 0.1)
        self.assertGreater(float(self.loss(attacked)), float(self.loss(self.pixels)))
        self.assertTrue(torch.allclose(attacked, torch.full_like(attacked, 0.4)))


if __name__ == "__main__":
    unittest.main()
