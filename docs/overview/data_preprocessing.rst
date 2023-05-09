Dataset configuration and data preprocessing
--------------------------------------------

Data preprocessing steps
^^^^^^^^^^^^^^^^^^^^^^^^

#. Data is fetched for the complete period between the train start and end time.
#. Data is then aggregated by ``aggregation_methods`` at given ``resolution``. 
   This step includes interpolating values that might be missing, handled in `Gordo <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/utils.py>`_.
   For this, the specified ``interpolation_method`` is used with its ``interpolation_limit``.
   The limit specifies how long from last valid data point values will be interpolated.
#. Known periods specified in ``known_filter_periods`` are dropped from dataset.
#. `Row filter <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/filter_rows.py>`_ is applied. 
   All numerical filtering criteria in the provided list are evaluated and joined by logical ``&``.
   This step also involves the ``row_filter_buffer_size``\ , which removes observations of the given size before and after the ones that have already been filtered out.
#. Data is filtered by global minima/maxima conditions (\ ``low_threshold``\ /\ ``high_threshold``\ ).
#. `Filter periods <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/filter_periods.py>`_ is applied for filtering out noisy data points.
    See below for some more details.
   If applied, the information is written to the `metadata <../metadata>`_ as ``filtered_periods``.

Between each step, the samples of the resulting dataset is compared to ``n_samples_threshold`` to ensure at least minimum size is returned.

Filter periods algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^

This executes a set of algorithms that will filter out observations.
There are currently two algorithms implemented: Rolling median and isolation forest.

``window`` is used by the rolling median 

.. list-table::
   :header-rows: 1

   * - Parameter
     - Method
     - Description
   * - ``window``
     - ``median``
     - Window for rolling median
   * - ``n_iqr``
     - ``median``
     - Number of interquartile (middle 50%) to add/subtract from rolling median
   * - ``iforest_smooth``
     - ``iforest``
     - If exponential weighted smoothing should be applied to data before isolation forest
   * - ``contamination``
     - ``iforest``
     - The amount of contamination of the data set, i.e. the proportion of outliers in the data set.


Dataset metadata
^^^^^^^^^^^^^^^^

The class `TimeSeriesDataset <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/datasets.py>`_ handles the metadata from the data.
Information regarding the size of each dataset, before any of the `preprocessing <../data_preprocessing>`_\ , as well as resamples and joined length of the dataset is generated `here <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/utils.py>`_. This information can be retrived with function ``get_metadata()``

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 30 70

   * - Path 
     - Description/Unit
   * - ``x_hist``
     - Histogram information for each tag. String with serialized JSON dictionary where key contains gap ``(<from>, <to>]``, and value number of values within this gap.
   * - ``data_provider``
     - Data provider specific metadata. 
   * - ``row_filter_tags``
     - List of sensor tags participating in raw filter `row filter <https://github.com/equinor/gordo-dataset/blob/master/gordo_dataset/filter_rows.py>`_.
   * - ``filtered_periods``
     - Periods dropped by applied algorithm in `data preprocessing <../data_preprocessing>`_
   * - ``summary_statistics``
     - Descriptive statistics (quartiles, max/min, median etc.) for each sensor tag
   * - ``tag_loading_metadata.tags``
     - Serialized sensor tags information.


``tag_loading_metadata.aggregate_metadata`` contains sensor tag lengths on the different preprocessing steps:

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 30 70

   * - Path 
     - Description/Unit
   * - ``joined_length``
     - Length after joining all sensor tags.
   * - ``dropped_na_length``
     - Length after dropping None values.


``tag_loading_metadata`` also contains per sensor tag metadata:

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 30 70

   * - Path 
     - Description/Unit
   * - ``gaps``
     - Time-series gaps with no values. ``start``, ``end`` of gaps in Unix-timestamp.
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
