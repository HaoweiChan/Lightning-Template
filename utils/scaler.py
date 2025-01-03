import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseScaler(ABC):
    """
    Base class for all scalers.
    
    All scalers should implement these methods to provide consistent
    scaling/unscaling functionality for PyTorch tensors.
    """
    def __init__(self):
        self.n_samples_seen_ = 0

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def fit(self, values):
        pass

    @abstractmethod
    def partial_fit(self, values):
        pass

    @abstractmethod
    def transform(self, values):
        pass

    @abstractmethod
    def fit_transform(self, values):
        pass

    @abstractmethod
    def inverse_transform(self, values):
        pass


class StandardScaler(BaseScaler):
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """
        Standard Scaler that normalizes tensors using mean and standard deviation.

        Args:
            mean: The mean of the features (set after calling fit)
            std: The standard deviation of the features (set after calling fit)
            epsilon: Small constant to avoid divide-by-zero errors
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def _reset(self):
        self.n_samples_seen_ = 0
        self.mean = None
        self.std = None

    def fit(self, values):
        self._reset()
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)
        self.n_samples_seen_ = values.shape[0]
        return self

    def transform(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * self.std.to(values.device) + self.mean.to(values.device)


class MinMaxScaler(BaseScaler):
    def __init__(self, min_=None, max_=None, ignore_value=None):
        """
        MinMax Scaler that scales tensors to a fixed range.

        Args:
            min_: Minimum value for scaling (set after calling fit)
            max_: Maximum value for scaling (set after calling fit)
            ignore_value: Value to ignore during scaling
        """
        super().__init__()
        self.min_ = min_
        self.max_ = max_
        self.ignore_value = ignore_value

    def _reset(self):
        self.n_samples_seen_ = 0
        if hasattr(self, "scale_"):
            del self.scale_
        if hasattr(self, "min_"):
            del self.min_
        if hasattr(self, "max_"):
            del self.max_

    def fit(self, values):
        self._reset()
        return self.partial_fit(values)

    def partial_fit(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)

        tensor_min = torch.where(values == self.ignore_value,
                               torch.tensor(float('inf')), values)
        tensor_max = torch.where(values == self.ignore_value,
                               torch.tensor(float('-inf')), values)

        data_min = torch.min(tensor_min, dim=0).values
        data_max = torch.max(tensor_max, dim=0).values

        if self.n_samples_seen_ == 0:
            self.min_ = data_min
            self.max_ = data_max
        else:
            self.min_ = torch.minimum(self.min_, data_min)
            self.max_ = torch.maximum(self.max_, data_max)

        self.n_samples_seen_ += len(values)
        self.scale_ = self.max_ - self.min_
        return self

    def transform(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)

        values_transformed = values.clone()
        values_transformed[values == self.ignore_value] = torch.nan

        valid_scale = torch.where(self.scale_ == 0.0,
                                torch.ones_like(self.scale_),
                                self.scale_)

        values_transformed = (values_transformed - self.min_) / valid_scale
        return torch.nan_to_num(values_transformed, nan=self.ignore_value)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * (self.max_.to(values.device) - self.min_.to(values.device)) \
               + self.min_.to(values.device)


class MaxAbsScaler(BaseScaler):
    def __init__(self, max_=None):
        """
        MaxAbs Scaler that scales tensors by their maximum absolute value.

        Args:
            max_: Maximum absolute value (set after calling fit)
        """
        super().__init__()
        self.max_ = max_

    def _reset(self):
        self.n_samples_seen_ = 0
        self.max_ = None

    def fit(self, values):
        self._reset()
        return self.partial_fit(values)

    def partial_fit(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)

        current_max = torch.max(torch.abs(values))
        if self.n_samples_seen_ == 0:
            self.max_ = current_max
        else:
            self.max_ = torch.max(self.max_, current_max)

        self.n_samples_seen_ += len(values)
        return self

    def transform(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)
        return values / (self.max_ + 1e-7)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * (self.max_ + 1e-7)
