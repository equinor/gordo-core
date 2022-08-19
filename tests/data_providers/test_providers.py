import pytest

from gordo_core.data_providers import providers
from gordo_core.data_providers.base import GordoBaseDataProvider


@pytest.mark.parametrize(
    "provider,expected_params",
    (
        (
            providers.RandomDataProvider(200, max_size=205),
            {"min_size": 200, "max_size": 205},
        ),
        (
            providers.InfluxDataProvider("measurement", value_name="Value"),
            {"measurement": "measurement", "value_name": "Value"},
        ),
    ),
)
def test_data_provider_serializations(
    provider: GordoBaseDataProvider, expected_params: dict
):
    """
    Test a given provider can be serialized to dict and back
    """

    encoded = provider.to_dict()

    # Verify the expected parameter kwargs match
    for k, v in expected_params.items():
        assert encoded[k] == v

    # Should have inserted the name of the class as 'type'
    assert provider.__module__ + "." + provider.__class__.__name__ == encoded["type"]

    # Should be able to recreate the object from encoded directly
    cloned = provider.__class__.from_dict(encoded)
    assert type(cloned) == type(provider)


def test_dummy_data_provider_serialization(dummy_data_provider):
    encoded = dummy_data_provider.to_dict()
    assert (
        dummy_data_provider.__class__.__module__
        + "."
        + dummy_data_provider.__class__.__name__
        == encoded["type"]
    )

    cloned = dummy_data_provider.from_dict(encoded)
    assert type(cloned) == type(dummy_data_provider)
