import copy
import itertools
import logging
from typing import Iterable, Optional, Union, cast

logger = logging.getLogger(__name__)


class SensorTagNormalizationError(ValueError):
    """Error indicating that something went wrong normalizing a sensor tag"""

    pass


class SensorTag:
    """
    Representation of sensor tag. It contains the sensor tag name and additional fields.

    Example
    -------
    >>> SensorTag("Test tag", field1="value1", field2="value2")
    SensorTag("Test tag",field1="value1",field2="value2")

    """

    def __init__(self, name: str, **kwargs: Optional[str]):
        """
        Parameters
        ----------
        name: str
            Sensor tag name. Required field.
        kwargs
            Additional fields.
        """
        self.name = name
        self._fields = kwargs
        self._validate()

    def _validate(self):
        for k, v in self._fields.items():
            if v is not None and type(v) is not str:
                raise SensorTagNormalizationError(
                    "Wrong value %s type for '%s'" % (repr(v), self.name)
                )

    def __getattr__(self, name):
        if "_fields" not in self.__dict__:
            raise AttributeError("Unable to get _fields")
        fields = self.__dict__["_fields"]
        if name in fields:
            return fields[name]
        class_name = self.__class__.__name__
        raise AttributeError("%s object has no attribute %s" % (class_name, name))

    def __eq__(self, other):
        if type(other) is not SensorTag:
            return False
        if self.name != other.name:
            return False
        return self._fields == other._fields

    def __contains__(self, item):
        if item == "name":
            return True
        return item in self._fields

    def __hash__(self):
        fields = tuple((k, v) for k, v in self._fields.items() if v is not None)
        return hash((self.name, fields))

    def __getstate__(self):
        return {
            "name": self.name,
            "fields": self._fields,
        }

    def __setstate__(self, state):
        self.name = state.get("name", "")
        self._fields = state.get("fields", {})

    def to_json(self) -> dict[str, Optional[str]]:
        """
        JSON representation of the sensor tag.

        Example
        -------
        >>> sensor_tag = SensorTag("Test tag", field1="value1", field2="value2")
        >>> sensor_tag.to_json()
        {'name': 'Test tag', 'field1': 'value1', 'field2': 'value2'}

        Returns
        -------
            Sensor tag name along with additional fields in one dict.
        """
        fields: dict[str, Optional[str]] = {"name": self.name}
        fields.update(self._fields)
        return fields

    def mutate_fields(self, **kwargs: Optional[str]) -> "SensorTag":
        """
        Update `SensorTag` fields from `kwargs`. Instantiate a new object if needed

        Examples
        --------
        >>> sensor_tag = SensorTag("Test tag", field1="value1", field2="value2")
        >>> sensor_tag.mutate_fields(field1="value", field3="value3")
        SensorTag("Test tag",field1="value",field2="value2",field3="value3")

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        new_fields = copy.copy(self._fields)
        new_fields.update(kwargs)
        if new_fields != self._fields:
            return SensorTag(self.name, **new_fields)
        else:
            return self

    def __repr__(self):
        fields = ['"' + self.name + '"']
        for k, v in self._fields.items():
            fields.append('%s="%s"' % (k, v))
        class_name = self.__class__.__name__
        return "%s(%s)" % (class_name, ",".join(fields))


Sensor = Union[dict[str, Optional[str]], str, SensorTag]
Tag = Union[str, SensorTag]


def _validate_required_fields(
    fields: dict,
    required_fields: Optional[Iterable[str]] = None,
    raise_exception: bool = True,
) -> bool:
    if type(fields) is not dict:
        raise SensorTagNormalizationError("Sensor %s has wrong type" % repr(fields))
    if not required_fields:
        return True
    for field in required_fields:
        if not fields.get(field, None):
            if raise_exception:
                raise SensorTagNormalizationError(
                    "Sensor representation %s does not have '%s'"
                    % (repr(fields), field)
                )
            return False
    return True


def load_sensor_tag(
    sensor: dict[str, Optional[str]],
    required_fields: Optional[Iterable[str]] = None,
) -> SensorTag:
    """
    Load sensor tag from the dict.

    Example
    -------
    >>> load_sensor_tag({"name": "tag", "field1": "value1"})
    SensorTag("tag",field1="value1")

    Parameters
    ----------
    sensor: dict[str, Optional[str]]
        It should at least contain `name` field.
    required_fields: Optional[Iterable[str]]
        Required additional fields.

    Returns
    -------
        SensorTag

    """
    _validate_required_fields(sensor, required_fields)
    sensor = copy.copy(sensor)
    name = sensor.pop("name", None)
    if not name:
        raise SensorTagNormalizationError(
            "Sensor representation %s does not have name" % repr(sensor)
        )
    return SensorTag(name, **sensor)


def normalize_sensor_tag(
    sensor: Sensor,
    required_fields: Optional[Iterable[str]] = None,
    **kwargs: Optional[str],
) -> Tag:
    """
    Take sensor tag information and tries to convert it to SensorTag.

    Parameters
    ----------
    sensor: Sensor
        Sensor tag information. Could be either dict suitable for `load_sensor_tag()` function or str
    required_fields: Optional[Iterable[str]]
        Required additional fields.
    kwargs
        Additional fields.

    Returns
    -------

    """
    sensor_tag: Tag

    if isinstance(sensor, SensorTag):
        sensor_tag = cast(SensorTag, sensor)
        _validate_required_fields(sensor_tag.to_json(), required_fields)

    elif isinstance(sensor, dict):
        sensor_tag = load_sensor_tag({**kwargs, **sensor}, required_fields)

    elif isinstance(sensor, str):
        if _validate_required_fields(kwargs, required_fields, raise_exception=False):
            sensor_tag = SensorTag(sensor, **kwargs)
        else:
            sensor_tag = sensor
    else:
        raise SensorTagNormalizationError(
            f"Sensor {sensor} with type {type(sensor)} cannot be converted to a valid "
            f"SensorTag"
        )

    return sensor_tag


def to_list_of_strings(sensor_tag_list: list[SensorTag]) -> list[str]:
    """
    List of sensor tags names.

    Parameters
    ----------
    sensor_tag_list: list[SensorTag]

    """
    return [sensor_tag.name for sensor_tag in sensor_tag_list]


def extract_tag_name(tag: Tag) -> str:
    """
    Get tag name.

    Parameters
    ----------
    tag: Tag

    """
    if type(tag) is str:
        return cast(str, tag)
    else:
        return cast(SensorTag, tag).name


def tag_to_json(tag: Tag) -> Union[str, dict[str, Optional[str]]]:
    """
    Convert `Tag` to JSON representation.

    Parameters
    ----------
    tag: Tag

    Returns
    -------

    """
    if type(tag) is str:
        return cast(str, tag)
    else:
        sensor_tag = cast(SensorTag, tag)
        return sensor_tag.to_json()


def validate_tag_equality(tag1: Tag, tag2: Tag):
    """
    SensorTag should not have a different asset name.
    str and SensorTag should not have the same name.

    Parameters
    ----------
    tag1: Tag
    tag2: Tag

    """
    type_tag1, type_tag2 = type(tag1), type(tag2)
    if type_tag1 is SensorTag and type_tag2 is SensorTag:
        if cast(SensorTag, tag1) != cast(SensorTag, tag2):
            raise ValueError(
                "Tags %s and %s with the same name but different fields"
                % (repr(tag1), repr(tag2))
            )
    for _type in (str, SensorTag):
        if type_tag1 is _type:
            if type_tag2 is not _type:
                tag_name1 = extract_tag_name(tag1)
                tag_name2 = extract_tag_name(tag2)
                if tag_name1 == tag_name2:
                    raise ValueError(
                        "Tags %s and %s has different type but the same name"
                        % (repr(tag1), repr(tag2))
                    )


def unique_tag_names(*tags: Iterable[Tag]) -> dict[str, Tag]:
    """
    Check the uniqueness of the tags.

    Parameters
    ----------
    tags
        Iterables of tags.

    Returns
    -------
    dict[str, Tag]
        Keys here are the unique tag names.
        Tags will be in lower case if `case_insensitive` True.

    """
    orig_tags: dict[str, Tag] = {}
    for tag in itertools.chain(*tags):
        tag_name = extract_tag_name(tag)
        if tag_name in orig_tags:
            validate_tag_equality(tag, orig_tags[tag_name])
        else:
            orig_tags[tag_name] = tag
    return orig_tags
