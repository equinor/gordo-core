from typing import Optional, Union


class InsufficientDataError(ValueError):
    """Generic error raised if the data is not enough to process."""


class EmptyDataframeError(InsufficientDataError):
    """Base error raised if the dataframe is empty / below threshold."""

    def __init__(self, data_length: int, threshold: int, *args, **kwargs):
        message = self._make_message(data_length, threshold, *args, **kwargs)

        super().__init__(message)

        self.data_length = data_length
        self.threshold = threshold

    @staticmethod
    def _make_message(data_length, threshold, *args, **kwargs):
        return (
            f"The length of the DataFrame ({ data_length }) does "
            + f"not exceed required threshold ({ threshold })."
        )


class EmptyGeneratedDataframeError(EmptyDataframeError):
    """Error raised if generated dataframe data is insufficient."""

    def __init__(
        self,
        data_length: int,
        threshold: int,
        tag_names: Optional[list[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(data_length, threshold, tag_names, *args, **kwargs)
        self.tag_names = tag_names

    @staticmethod
    def _make_message(
        data_length, threshold, tag_names: Optional[list[str]] = None, *args, **kwargs
    ):
        if tag_names:
            tag_names_message = ", ".join(tag_names)
            message_tail = f"Tags with no points: {tag_names_message}."
        else:
            message_tail = "No metadata with empty tags available."

        return (
            f"The length of the generated DataFrame ({ data_length }) does "
            + f"not exceed required threshold ({ threshold }).  { message_tail }"
        )


class EmptyFilteredDataframeError(EmptyDataframeError):
    """Base error raised if row filtering made the data insufficient."""

    _message_tail = "."

    @classmethod
    def _make_message(cls, data_length, threshold, *args, **kwargs):
        return (
            f"The length of the filtered DataFrame ({ data_length }) does "
            + f"not exceed required threshold ({ threshold }){cls._message_tail}"
        )


class RowFilterEmptyDataError(EmptyFilteredDataframeError):
    """Error raised if row filtering made the data insufficient."""

    def __init__(
        self,
        data_length: int,
        threshold: int,
        row_filter: Union[str, list[str]],
        row_filter_buffer_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            data_length, threshold, row_filter, row_filter_buffer_size, *args, **kwargs
        )
        self.row_filter = row_filter
        self.row_filter_buffer_size = row_filter_buffer_size

    @classmethod
    def _make_message(
        cls, data_length, threshold, row_filter, row_filter_buffer_size, *args, **kwargs
    ):
        base_message = super()._make_message(data_length, threshold, *args, **kwargs)

        return (
            base_message
            + f"  Applied the row filter: {repr(row_filter)} "
            + f"with buffer size: {repr(row_filter_buffer_size)}."
        )


class KnownPeriodsEmptyDataError(EmptyFilteredDataframeError):
    """Error raised if known periods filter made the data insufficient."""

    _message_tail = " after dropping known periods."


class GlobalExtremaEmptyDataError(EmptyFilteredDataframeError):
    """Error raised if global extrema filter made the data insufficient."""

    _message_tail = " after filtering global extrema."


class NuisanceEmptyDataError(EmptyFilteredDataframeError):
    """Error raised if nuisance filter made the data insufficient."""

    _message_tail = " after applying nuisance filtering algorithm."


class ConfigException(ValueError):
    """Generic error raised if the config is not valid."""
