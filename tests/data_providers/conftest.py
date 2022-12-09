from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import pytest

from gordo_core.data_providers.base import GordoBaseDataProvider
from gordo_core.sensor_tag import Tag
from gordo_core.utils import capture_args


class DummyDataProvider(GordoBaseDataProvider):
    @capture_args
    def __init__(self, arg1) -> None:
        self.arg1 = arg1

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: list[Tag],
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> Iterable[pd.Series]:
        yield pd.Series()

    def to_dict(self):
        if not hasattr(self, "_params"):
            raise AttributeError(
                "Failed to lookup init parameters, ensure the "
                "object's __init__ is decorated with 'capture_args'"
            )
        # Update dict with the class
        params = getattr(self, "_params", {})
        module_str = self.__class__.__module__
        if module_str is None or module_str == str.__class__.__module__:
            module_str = self.__class__.__name__
        else:
            module_str = module_str + "." + self.__class__.__name__
        params["type"] = module_str
        return params


@pytest.fixture
def dummy_data_provider():
    return DummyDataProvider("test_arg")
