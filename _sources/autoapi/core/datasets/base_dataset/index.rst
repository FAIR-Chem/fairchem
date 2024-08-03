core.datasets.base_dataset
==========================

.. py:module:: core.datasets.base_dataset

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.datasets.base_dataset.T_co


Exceptions
----------

.. autoapisummary::

   core.datasets.base_dataset.UnsupportedDatasetError


Classes
-------

.. autoapisummary::

   core.datasets.base_dataset.DatasetMetadata
   core.datasets.base_dataset.BaseDataset
   core.datasets.base_dataset.Subset


Functions
---------

.. autoapisummary::

   core.datasets.base_dataset.create_dataset


Module Contents
---------------

.. py:data:: T_co

.. py:class:: DatasetMetadata

   Bases: :py:obj:`NamedTuple`


   .. py:attribute:: natoms
      :type:  numpy.typing.ArrayLike | None
      :value: None



.. py:exception:: UnsupportedDatasetError

   Bases: :py:obj:`ValueError`


   Inappropriate argument value (of correct type).


.. py:class:: BaseDataset(config: dict)

   Bases: :py:obj:`torch.utils.data.Dataset`\ [\ :py:obj:`T_co`\ ]


   Base Dataset class for all OCP datasets.


   .. py:attribute:: config


   .. py:attribute:: paths
      :value: []



   .. py:attribute:: lin_ref
      :value: None



   .. py:method:: __len__() -> int


   .. py:method:: metadata_hasattr(attr) -> bool


   .. py:property:: indices


   .. py:property:: _metadata
      :type: DatasetMetadata



   .. py:method:: get_metadata(attr, idx)


.. py:class:: Subset(dataset: BaseDataset, indices: collections.abc.Sequence[int], metadata: DatasetMetadata | None = None)

   Bases: :py:obj:`torch.utils.data.Subset`, :py:obj:`BaseDataset`


   A pytorch subset that also takes metadata if given.


   .. py:attribute:: metadata


   .. py:attribute:: indices


   .. py:attribute:: num_samples


   .. py:attribute:: config


   .. py:property:: _metadata
      :type: DatasetMetadata



   .. py:method:: get_metadata(attr, idx)


.. py:function:: create_dataset(config: dict[str, Any], split: str) -> Subset

   Create a dataset from a config dictionary

   :param config: dataset config dictionary
   :type config: dict
   :param split: name of split
   :type split: str

   :returns: dataset subset class
   :rtype: Subset


