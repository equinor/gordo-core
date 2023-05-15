Data source customization
-------------------------

To extend ``GordoBaseDataProvider`` class for a custom data source, you will need to create a new class that inherits from the ``GordoBaseDataProvider`` class. Override the ``get_data()`` method to retrieve data from your custom data source and return it in the desired format.

As an example, let's make a CSV reader ``CSVDataProvider``.

.. literalinclude:: ../../gordo_core/data_providers/examples/csv_provider.py

Then we can use this data provider through ``TimeSeriesDataset`` and load a CSV file.

.. ipython::

    In [1]: from gordo_core.time_series import TimeSeriesDataset

    In [2]: from gordo_core.data_providers.examples.csv_provider import CSVDataProvider

    In [3]: data_provider=CSVDataProvider("../examples/turbine_sensors.csv", "index")

    In [4]: dataset = TimeSeriesDataset(
       ...:     train_start_date='2023-01-29 00:00:00+00:00',
       ...:     train_end_date='2023-01-31 00:00:00+00:00',
       ...:     tag_list=['Pressure','RPM','Temperature'],
       ...:     data_provider=data_provider,
       ...:     row_filter="`RPM` > 0",
       ...: )

    In [5]: X, y = dataset.get_data()

    In [6]: X

