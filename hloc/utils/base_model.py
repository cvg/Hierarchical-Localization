import inspect
import sys
from abc import ABCMeta, abstractmethod
from copy import copy
from types import ModuleType
from typing import Any

from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf: dict[str, Any] = {}
    required_inputs: list[str] = []

    def __init__(self, conf: dict[str, Any]) -> None:
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, f"Missing key {key} in data"
        return self._forward(data)

    @abstractmethod
    def _init(self, conf: dict[str, Any]) -> None:
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """To be implemented by the child class."""
        raise NotImplementedError


def dynamic_load(root: ModuleType, model: str) -> type[BaseModel]:
    module_path = f"{root.__name__}.{model}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    model_classes = [cls for _, cls in classes if issubclass(cls, BaseModel)]
    assert len(model_classes) == 1, model_classes
    return model_classes[0]
    # return getattr(module, 'Model')
