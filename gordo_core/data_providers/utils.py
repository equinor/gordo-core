from typing import Iterable, Tuple
from urllib.parse import quote

from gordo_core.file_systems import FileSystem


def partition_dir_name(field: str, value: str):
    return field + "=" + quote(value, safe=" ")


def build_dir_path(
    storage: FileSystem, base_dir: str, field_values: Iterable[Tuple[str, str]]
) -> str:
    dir_path = base_dir
    for field, value in field_values:
        dir_path = storage.join(dir_path, partition_dir_name(field, value))
    return dir_path
