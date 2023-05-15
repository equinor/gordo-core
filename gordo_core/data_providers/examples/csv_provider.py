import pandas as pd

from pathlib import Path
from datetime import datetime

from gordo_core.data_providers.base import GordoBaseDataProvider
from gordo_core.utils import capture_args
from gordo_core.sensor_tag import unique_tag_names, Tag

from typing import Optional, Union, Iterable, Tuple


class CSVDataProvider(GordoBaseDataProvider):
    @capture_args
    def __init__(
        self,
        file_path: Union[str, Path],
        timestamp_column: str,
        sep: str = ",",
        delimiter: Optional[str] = None,
    ):
        self.file_path = file_path
        self.timestamp_column = timestamp_column
        self.sep = sep
        self.delimiter = delimiter

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: list[Tag],
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> Iterable[Tuple[pd.Series, Tag]]:
        tags = unique_tag_names(tag_list)
        usecols = []
        if self.timestamp_column not in tags:
            usecols.append(self.timestamp_column)
        usecols.extend(tags.keys())
        df = pd.read_csv(
            self.file_path,
            sep=self.sep,
            delimiter=self.delimiter,
            usecols=usecols,
        )
        df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], utc=True)
        filtered = df[
            (df[self.timestamp_column] >= train_start_date)
            & (df[self.timestamp_column] < train_end_date)
        ]
        filtered = filtered.set_index(self.timestamp_column)
        for column in filtered:
            yield filtered[column], tags[column]
