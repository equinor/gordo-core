import posixpath
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import IO, Iterable, Optional, Tuple


class FileType(Enum):
    """
    Type of file system item.
    """

    DIRECTORY = 1
    FILE = 2


@dataclass(frozen=True)
class FileInfo:
    """
    File/directory information.
    """

    file_type: FileType
    size: int
    """
    Size of the file.
    """
    access_time: Optional[datetime] = None
    """
    Time of last access.
    """
    modify_time: Optional[datetime] = None
    """
    Time of last modification.
    """
    create_time: Optional[datetime] = None
    """
    Time of creation.
    """

    def isfile(self) -> bool:
        return self.file_type == FileType.FILE

    def isdir(self) -> bool:
        return self.file_type == FileType.DIRECTORY


def default_join(*p) -> str:
    if p and not p[-1]:
        p = p[:-1]
    if not p:
        return ""
    return posixpath.join(*p)


class FileSystem(metaclass=ABCMeta):
    """
    An interface for file system implementations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this file system. Example: ``S3://mybucket/home``
        """
        ...

    @abstractmethod
    def open(self, path: str, mode: str = "r") -> IO:
        """
        Open a file. :func:`open` analog for this file system.

        Parameters
        ----------
        path
            Path to the file
        mode
            Required support at least of `r`,`b` modes
        """
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Tests whether a path exists.
        """
        ...

    @abstractmethod
    def isfile(self, path: str) -> bool:
        """
        Tests whether a path is a regular file.
        """
        ...

    @abstractmethod
    def isdir(self, path: str) -> bool:
        """
        Returns true if ``path`` refers to an existing directory.
        """
        ...

    @abstractmethod
    def info(self, path: str) -> FileInfo:
        """
        Retrieves :class:`.FileInfo` of a path.
        """
        ...

    @abstractmethod
    def ls(
        self, path: str, with_info: bool = True
    ) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        """
        Returns a list containing the names of the files in the directory.

        Parameters
        ----------
        path
            Directory path.
        with_info
            Retrieves :class:`.FileInfo` for each item. Otherwise, will be ``None`` for all items.
        """
        ...

    @abstractmethod
    def walk(
        self, base_path: str, with_info: bool = True
    ) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        """
        Directory tree recursive iterator.

        Parameters
        ----------
        base_path
            Directory path.
        with_info
            :class:`~FileSystem.ls` ``with_info`` argument analog.

        Returns
        -------
        The first item of ``tuple`` is a path directory tree item, the second is :class:`.FileInfo` if ``with_info=True``
        """
        ...

    def join(self, *p) -> str:
        """
        Join two or more pathname components, specific for this file system.

        Parameters
        ----------
        p
            Path components list.
        """
        return default_join(*p)

    def split(self, p: str) -> Tuple[str, str]:
        """
        Split a pathname.
        """
        return posixpath.split(p)
