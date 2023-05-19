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

.. _filter_periods:

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

.. toctree::
    :maxdepth: 1
