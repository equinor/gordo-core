import importlib
from typing import Any, Iterable, Optional

Location = tuple[Optional[str], str]
BackCompatibleLocations = dict[Location, Location]


def _parse_str_location(location: str) -> Location:
    if "." in location:
        module_name, class_name = location.rsplit(".", 1)
        return module_name, class_name
    else:
        return None, location


def _get_back_compatible_location(
    location: Location, back_compatibles: BackCompatibleLocations
) -> Location:
    current_location = location
    visited_locations: set[Location] = {current_location}
    while current_location in back_compatibles:
        current_location = back_compatibles[current_location]
        if current_location in visited_locations:
            raise RecursionError(
                "Found recursion in BackCompatibleLocations. Location: %s"
                % repr(location)
            )
        visited_locations.add(current_location)
    return current_location


def prepare_back_compatible_locations(
    locations: Iterable[tuple[str, str]]
) -> BackCompatibleLocations:
    """
    The result of this function can be used as `back_compatibles` argument
    in `import_location()` functions.

    Example
    -------
    >>> prepare_back_compatible_locations([('old_module.MyClass', 'new_module.MyClass'),('OldClass', 'NewClass')])
    {('old_module', 'MyClass'): ('new_module', 'MyClass'), (None, 'OldClass'): (None, 'NewClass')}

    Parameters
    ----------
    locations: Iterable[tuple[str, str]]
        List of locations. The first item of each tuple is a location in the previous version,
        the second item is the location of the current version.

    Returns
    -------
        Key-Value pair with locations of the previous version to the current version.

    """
    back_compatibles: BackCompatibleLocations = {}
    for from_location, to_location in locations:
        key = _parse_str_location(from_location)
        if key in back_compatibles:
            raise ValueError("Duplicate locations: '%s'" % from_location)
        value = _parse_str_location(to_location)
        back_compatibles[key] = value
    # Validate for recursions
    for location in back_compatibles.keys():
        _get_back_compatible_location(location, back_compatibles)
    return back_compatibles


def import_location(
    location: str,
    *,
    import_path: Optional[str] = None,
    back_compatibles: Optional[BackCompatibleLocations] = None
) -> Any:
    """
    Imports entity from provided `location`, or finds an entity with `location` name in `import_path` module.

    Example
    -------
    >>> import_location("multiprocessing.Process")
    <class 'multiprocessing.context.Process'>
    >>> import_location("Process", import_path="multiprocessing")
    <class 'multiprocessing.context.Process'>

    Parameters
    ----------
    location: str
    import_path: Optional[str]
    back_compatibles: Optional[BackCompatibleLocations]
        See `prepare_back_compatible_locations()` function for reference.

    Returns
    -------
        Imported entity.
    """
    parsed_location = _parse_str_location(location)
    if back_compatibles is not None:
        parsed_location = _get_back_compatible_location(
            parsed_location, back_compatibles
        )
    module_name, entity_name = parsed_location
    if module_name is None:
        if import_path is None:
            raise ImportError(
                "Unable to import '%s'. import_path should be provided" % location
            )
        module_name = import_path
    module = importlib.import_module(module_name)
    if not hasattr(module, entity_name):
        raise ImportError(
            "Unable to find '%s' in module '%s'" % (entity_name, module_name)
        )
    return getattr(module, entity_name)
