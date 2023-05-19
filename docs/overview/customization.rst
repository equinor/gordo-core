Data source customization
-------------------------

For implementing a custom data source extend :class:`gordo_core.data_providers.base.GordoBaseDataProvider`.
Override :func:`gordo_core.data_providers.base.GordoBaseDataProvider.load_series` method, it should return data from
the data source in a correct format.

As a reference we could use CSV reader from :mod:`gordo_core.data_providers.contrib` module:

.. literalinclude:: ../../gordo_core/data_providers/contrib/csv_provider.py

Then use this data provider with :class:`gordo_core.time_series.TimeSeriesDataset` to load a CSV file:

.. ipython::

    In [1]: from gordo_core.time_series import TimeSeriesDataset

    In [2]: from gordo_core.sensor_tag import SensorTag 

    In [3]: from gordo_core.data_providers.contrib.csv_provider import CSVDataProvider

    In [4]: data_provider=CSVDataProvider("../examples/turbine_sensors.csv", "index")

    In [5]: dataset = TimeSeriesDataset(
       ...:     train_start_date='2023-01-29 00:00:00+00:00',
       ...:     train_end_date='2023-01-31 00:00:00+00:00',
       ...:     tag_list=[SensorTag('Pressure'), SensorTag('RPM'), 'Temperature'],
       ...:     data_provider=data_provider,
       ...:     row_filter="`RPM` > 0",
       ...: )

    In [6]: X, y = dataset.get_data()

    In [7]: X

``tag_list`` could be specified either as :class:`gordo_core.sensor_tag.SensorTag` object with additional metadata or as a string.
``str`` to :class:`gordo_core.sensor_tag.SensorTag` conversion should be customized with overwriting
:func:`gordo_core.data_providers.base.GordoBaseDataProvider.tag_normalizer` method.

