import pandas as pd

from pathlib import Path
from datetime import datetime

from gordo_core.data_providers.base import GordoBaseDataProvider
from gordo_core.utils import capture_args
from gordo_core.sensor_tag import unique_tag_names, Tag

from typing import Optional, Union, Iterable, Tuple


class CSVDataProvider(GordoBaseDataProvider):
    @capture_args  # required for proper data provider JSON serialization
    def __init__(
        self, file_path: Union[str, Path], timestamp_column: str, sep: str = ","
    ):
        """
        Parameters
        ----------
        file_path
            Path to a CSV file containing the data to be loaded.
        timestamp_column
            Column in the CSV file containing the timestamps for each row.
        sep
            Delimiter to use.
        """
        self.file_path = file_path
        self.timestamp_column = timestamp_column
        self.sep = sep

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: list[Tag],
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> Iterable[Tuple[pd.Series, Tag]]:
        """
        Load the data from the CSV file.
        """
        # this dict contains sensor tag names as keys, and Tag as values
        tags = unique_tag_names(tag_list)
        usecols = []
        if self.timestamp_column not in tags:
            usecols.append(self.timestamp_column)
        usecols.extend(tags.keys())
        df = pd.read_csv(
            self.file_path,
            sep=self.sep,
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
