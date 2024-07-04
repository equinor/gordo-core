"""Module with expected metadata of joined to Dataframe Series."""

__all__ = ["get_expected_metadata"]


def _tags_section(first_tag_name: str, second_tag_name: str) -> dict:
    return {
        "tags": {
            first_tag_name: {"name": first_tag_name},
            second_tag_name: {"name": second_tag_name},
        },
    }


def tc1_expected_metadata(first_tag_name: str, second_tag_name: str) -> dict:
    return {
        first_tag_name: {
            "original_length": 126,
            "resampled_length": 145,
            "first_timestamp": 1628726400.0,
            "last_timestamp": 1628812800.0,
            "gaps": {},
        },
        second_tag_name: {
            "original_length": 334,
            "resampled_length": 145,
            "first_timestamp": 1628726400.0,
            "last_timestamp": 1628812800.0,
            "gaps": {},
        },
        "aggregate_metadata": {"joined_length": 145, "dropped_na_length": 145},
        **_tags_section(first_tag_name, second_tag_name),
    }


def tc2_expected_metadata(first_tag_name: str, second_tag_name: str) -> dict:
    return {
        first_tag_name: {
            "original_length": 58,
            "resampled_length": 126,
            "first_timestamp": 1634267400.0,
            "last_timestamp": 1634342400.0,
            "gaps": {"start": [1634256000.0], "end": [1634267400.0]},
        },
        second_tag_name: {
            "original_length": 310,
            "resampled_length": 145,
            "first_timestamp": 1634256000.0,
            "last_timestamp": 1634342400.0,
            "gaps": {},
        },
        "aggregate_metadata": {"joined_length": 126, "dropped_na_length": 126},
        **_tags_section(first_tag_name, second_tag_name),
    }


def tc3_expected_metadata(first_tag_name: str, second_tag_name: str) -> dict:
    return {
        first_tag_name: {
            "original_length": 206,
            "resampled_length": 145,
            "first_timestamp": 1631750400.0,
            "last_timestamp": 1631836800.0,
            "gaps": {},
        },
        second_tag_name: {
            "original_length": 206,
            "resampled_length": 145,
            "first_timestamp": 1631750400.0,
            "last_timestamp": 1631836800.0,
            "gaps": {},
        },
        "aggregate_metadata": {"joined_length": 145, "dropped_na_length": 145},
        **_tags_section(first_tag_name, second_tag_name),
    }


def tc4_expected_metadata(first_tag_name: str, second_tag_name: str) -> dict:
    return {
        first_tag_name: {
            "original_length": 206,
            "resampled_length": 145,
            "first_timestamp": 1631750400.0,
            "last_timestamp": 1631836800.0,
            "gaps": {},
        },
        second_tag_name: {"original_length": 206, "resampled_length": 0},
        "aggregate_metadata": {"joined_length": 0, "dropped_na_length": 0},
        **_tags_section(first_tag_name, second_tag_name),
    }


def tc5_expected_metadata(first_tag_name: str, second_tag_name: str) -> dict:
    return {
        first_tag_name: {
            "original_length": 56,
            "resampled_length": 145,
            "first_timestamp": 1641996000.0,
            "last_timestamp": 1642082400.0,
            "gaps": {},
        },
        second_tag_name: {
            "original_length": 320,
            "resampled_length": 145,
            "first_timestamp": 1641996000.0,
            "last_timestamp": 1642082400.0,
            "gaps": {},
        },
        "aggregate_metadata": {"joined_length": 145, "dropped_na_length": 145},
        **_tags_section(first_tag_name, second_tag_name),
    }


def tc6_expected_metadata(first_tag_name: str, second_tag_name: str) -> dict:
    return {
        first_tag_name: {
            "original_length": 206,
            "resampled_length": 145,
            "first_timestamp": 1631750400.0,
            "last_timestamp": 1631836800.0,
            "gaps": {},
        },
        second_tag_name: {
            "original_length": 206,
            "resampled_length": 1,
            "first_timestamp": 1631750400.0,
            "last_timestamp": 1631750400.0,
            "gaps": {"start": [1631750400.0], "end": [1631836800.0]},
        },
        "aggregate_metadata": {"joined_length": 1, "dropped_na_length": 1},
        **_tags_section(first_tag_name, second_tag_name),
    }


def tc7_expected_metadata(first_tag_name: str, second_tag_name: str) -> dict:
    return {
        first_tag_name: {
            "original_length": 56,
            "resampled_length": 55,
            "first_timestamp": 1642022400.0,
            "last_timestamp": 1642054800.0,
            "gaps": {},
        },
        second_tag_name: {
            "original_length": 97,
            "resampled_length": 55,
            "first_timestamp": 1642022400.0,
            "last_timestamp": 1642054800.0,
            "gaps": {},
        },
        "aggregate_metadata": {"joined_length": 55, "dropped_na_length": 55},
        **_tags_section(first_tag_name, second_tag_name),
    }


def get_expected_metadata(test_id: str, first_tag_name: str, second_tag_name: str):
    """Get metadata for specific test case number.

    'test_id' is ID of test on what function's name should start.
    """
    metadata_functions = [
        tc1_expected_metadata,
        tc2_expected_metadata,
        tc3_expected_metadata,
        tc4_expected_metadata,
        tc5_expected_metadata,
        tc6_expected_metadata,
        tc7_expected_metadata,
    ]

    for f in metadata_functions:
        if f.__name__.startswith(test_id):
            return f(first_tag_name, second_tag_name)
    raise ValueError(f"Function to make metadata for '{test_id}' is not defined.")
