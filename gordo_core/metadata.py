from typing import Iterable, Optional, Set

from gordo_core.sensor_tag import (
    SensorTag,
    Tag,
    extract_tag_name,
    load_sensor_tag,
    tag_to_json,
    unique_tag_names,
)


def tags_to_json_representation(tags: Iterable[Tag]) -> dict:
    unique_tags = unique_tag_names(tags)
    tags_metadata = {}
    for tag_name, tag in unique_tags.items():
        json_repr = tag_to_json(tag)
        if type(json_repr) is str:
            continue
        tags_metadata[tag_name] = json_repr
    return tags_metadata


def _get_dataset_meta(build_dataset_metadata: dict) -> Optional[dict]:
    if "dataset_meta" in build_dataset_metadata:
        return build_dataset_metadata["dataset_meta"]
    return None


def _tags_from_build_metadata(build_dataset_metadata: dict) -> Optional[dict]:
    dataset_meta = _get_dataset_meta(build_dataset_metadata)
    if dataset_meta is not None:
        if "tag_loading_metadata" in dataset_meta:
            tag_loading_metadata = dataset_meta["tag_loading_metadata"]
            return tag_loading_metadata.get("tags")
    return None


_list_of_tags_exception_message = (
    "The list of tags should be placed on"
    " dataset.dataset_meta.tag_loading_metadata.tags path"
)


def sensor_tags_from_build_metadata(
    build_dataset_metadata: dict, tag_names: Set[str]
) -> dict[str, SensorTag]:
    """
    Fetch tags assets from the metadata

    Parameters
    ----------
    build_dataset_metadata: dict
        build_metadata.dataset part of the metadata
    tag_names: Set[str]
        Contains tag names for which we should fetch information
    Returns
    -------
    dict[str, SensorTag]
        Key here is tag name passed though `tag_names` argument

    """
    tags_build_metadata = _tags_from_build_metadata(build_dataset_metadata)
    sensor_tags: dict[str, SensorTag] = {}
    if tags_build_metadata is None:
        raise ValueError(
            "Unable to find tags information in build_metadata. "
            + _list_of_tags_exception_message
        )
    for tag_name in tag_names:
        if tag_name not in tags_build_metadata:
            raise ValueError(
                "Unable to find tag '%s' information in build_metadata. "
                + _list_of_tags_exception_message
            )
        tag_metadata = tags_build_metadata[tag_name]
        sensor_tag = load_sensor_tag(tag_metadata)
        sensor_tags[sensor_tag.name] = sensor_tag
    return sensor_tags
