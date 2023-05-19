Metadata
--------

The class `TimeSeriesDataset <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/datasets.py>`_ handles
the metadata and some metrics from the loaded data.
Information regarding the size of each dataset, before any of the `preprocessing <../data_preprocessing>`_\ ,
as well as resamples and joined length of the dataset is generated `here <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/utils.py>`_.
This information can be retrived with function ``get_metadata()``.

.. list-table::
   :header-rows: 1
   :width: 100%

   * - Path
     - Description/Unit
   * - ``x_hist``
     - Histogram information for each tag. String with serialized JSON dictionary where key contains gap ``(<from>, <to>]``, and value number of samples within this gap.
   * - ``data_provider``
     - Data provider specific metadata. Result of ``GordoBaseDataProvider.get_metadata()`` method.
   * - ``row_filter_tags``
     - List of row filter tags. Tags participating in `Row filter <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/filter_rows.py>`_.
   * - ``filtered_periods``
     - Periods dropped by applied algorithm in the data preprocessing. Key is a :ref:`filter periods <filter_periods>` type, and value is a list with ``drop_start``, ``drop_end`` items in ISO timestamps.
   * - ``summary_statistics``
     - Descriptive statistics (quartiles, max/min, median etc.) for each tag
   * - ``tag_loading_metadata.tags``
     - Serialized tags information.
   * - ``tag_loading_metadata.aggregate_metadata``
     - DataFrame sizes: ``joined_length`` - after joining all columns,  ``dropped_na_length`` - after removing missing values.

``tag_loading_metadata`` also contains per tag metadata:

.. list-table::
   :header-rows: 1
   :width: 100%

   * - Path
     - Description/Unit
   * - ``gaps``
     - Empty time-series gaps. ``start``, ``end`` of gaps in Unix-timestamp.
   * - ``last_timestamp``
     - First value in Unix-timestamp.
   * - ``first_timestamp``
     - Last value in Unix-timestamp.
   * - ``original_length``
     - Number of values.
   * - ``resampled_length``
     - Number of values after resampling.

.. toctree::
    :maxdepth: 1
