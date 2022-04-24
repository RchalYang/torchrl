"""Normalizer."""
import numpy as np
import torch


def update_mean_var_count(
        mean, var, count,
        batch_mean, batch_var, batch_count):
  """
  Imported From OpenAI Baseline
  """
  delta = batch_mean - mean
  tot_count = count + batch_count

  new_mean = mean + delta * batch_count / tot_count
  m_a = var * count
  m_b = batch_var * batch_count
  m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
  new_var = m2 / tot_count
  new_count = tot_count

  return new_mean, new_var, new_count


class Normalizer():
  """Normalizer for observation normalization and advantage normalization"""

  def __init__(self, shape, clip=10.):
    self.shape = shape
    self._mean = np.zeros(shape)
    self._var = np.ones(shape)
    self._count = 1e-4
    self.clip = clip
    self.should_estimate = True

  def stop_update_estimate(self):
    self.should_estimate = False

  def update_estimate(self, data):
    if not self.should_estimate:
      return
    if len(data.shape) == self.shape:
      data = data[np.newaxis, :]
    self._mean, self._var, self._count = update_mean_var_count(
        self._mean, self._var, self._count,
        np.mean(data, axis=0), np.var(data, axis=0), data.shape[0])

  def inverse(self, raw):
    return raw * np.sqrt(self._var) + self._mean

  def filt(self, raw):
    return np.clip(
        (raw - self._mean) / (np.sqrt(self._var) + 1e-4),
        -self.clip, self.clip)


def update_mean_var_count_torch(
        mean, var, count,
        batch_mean, batch_var, batch_count):
  """
  Imported From OpenAI Baseline
  """
  delta = batch_mean - mean
  tot_count = count + batch_count

  new_mean = mean + delta * batch_count / tot_count
  m_a = var * count
  m_b = batch_var * batch_count
  m2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
  new_var = m2 / tot_count
  new_count = tot_count

  return new_mean, new_var, new_count


class TorchNormalizer(Normalizer):
  """Torch Version Normalizer."""

  def __init__(self, shape, device, clip=10.):
    super().__init__(shape, clip=10.)
    self.shape = shape
    self.device = device
    self._mean = torch.zeros(shape).to(self.device)
    self._var = torch.ones(shape).to(self.device)
    self._count = 1e-4
    self.clip = clip
    self.should_estimate = True

  def stop_update_estimate(self):
    self.should_estimate = False

  def update_estimate(self, data):
    if not self.should_estimate:
      return
    if len(data.shape) == self.shape:
      data = data.unsqueeze(0)
    self._mean, self._var, self._count = update_mean_var_count_torch(
        self._mean, self._var, self._count,
        torch.mean(data, dim=0), torch.var(data, dim=0), data.shape[0]
    )

  def inverse(self, raw):
    return raw * torch.sqrt(self._var) + self._mean

  def filt(self, raw):
    return torch.clamp(
        (raw - self._mean) / (torch.sqrt(self._var) + 1e-4),
        -self.clip, self.clip
    )
