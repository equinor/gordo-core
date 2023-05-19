.. gordo-core documentation master file, created by
   sphinx-quickstart on Sat May  6 12:08:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gordo-core
==========

gordo-core can fetch data from various sources, including databases, APIs, and file systems. The library provides a unified interface for accessing these data sources, making integrating new sources into the workflow easy.
It also provides tools for preprocessing and cleaning fetched data.

Example of usage:

.. ipython::
    :okwarning:

    In [1]: from gordo_core.time_series import RandomDataset

    In [2]: dataset = RandomDataset(
       ...:     train_start_date='2023-01-01 00:00:00+00:00',
       ...:     train_end_date='2023-01-31 00:00:00+00:00',
       ...:     tag_list=['tag1', 'tag2', 'tag3']
       ...: )

    In [3]: X, y = dataset.get_data()

    In [4]: X

.. toctree::
    :maxdepth: 2

    ./overview/data_preprocessing.rst
    ./overview/customization.rst
    ./api/index.rst
