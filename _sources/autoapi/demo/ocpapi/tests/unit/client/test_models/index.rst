:py:mod:`demo.ocpapi.tests.unit.client.test_models`
===================================================

.. py:module:: demo.ocpapi.tests.unit.client.test_models


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   demo.ocpapi.tests.unit.client.test_models.ModelTestWrapper
   demo.ocpapi.tests.unit.client.test_models.TestModel
   demo.ocpapi.tests.unit.client.test_models.TestModels
   demo.ocpapi.tests.unit.client.test_models.TestBulk
   demo.ocpapi.tests.unit.client.test_models.TestBulks
   demo.ocpapi.tests.unit.client.test_models.TestAdsorbates
   demo.ocpapi.tests.unit.client.test_models.TestAtoms
   demo.ocpapi.tests.unit.client.test_models.TestSlabMetadata
   demo.ocpapi.tests.unit.client.test_models.TestSlab
   demo.ocpapi.tests.unit.client.test_models.TestSlabs
   demo.ocpapi.tests.unit.client.test_models.TestAdsorbateSlabConfigs
   demo.ocpapi.tests.unit.client.test_models.TestAdsorbateSlabRelaxationsSystem
   demo.ocpapi.tests.unit.client.test_models.TestAdsorbateSlabRelaxationsRequest
   demo.ocpapi.tests.unit.client.test_models.TestAdsorbateSlabRelaxationsRequest_req_fields_only
   demo.ocpapi.tests.unit.client.test_models.TestAdsorbateSlabRelaxationResult
   demo.ocpapi.tests.unit.client.test_models.TestAdsorbateSlabRelaxationResult_req_fields_only
   demo.ocpapi.tests.unit.client.test_models.TestAdsorbateSlabRelaxationsResults




Attributes
~~~~~~~~~~

.. autoapisummary::

   demo.ocpapi.tests.unit.client.test_models.T


.. py:data:: T

   

.. py:class:: ModelTestWrapper


   .. py:class:: ModelTest(*args: Any, obj: T, obj_json: str, **kwargs: Any)


      Bases: :py:obj:`unittest.TestCase`, :py:obj:`Generic`\ [\ :py:obj:`T`\ ]

      Base class for all tests below that assert behavior of data models.

      .. py:method:: test_from_json() -> None


      .. py:method:: test_to_json() -> None


      .. py:method:: assertJsonEqual(first: str, second: str) -> None

         Compares two JSON-formatted strings by deserializing them and then
         comparing the generated built-in types.




.. py:class:: TestModel(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.Model`\ ]

   Serde tests for the Model data model.


.. py:class:: TestModels(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.Models`\ ]

   Serde tests for the Models data model.


.. py:class:: TestBulk(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.Bulk`\ ]

   Serde tests for the Bulk data model.


.. py:class:: TestBulks(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.Bulks`\ ]

   Serde tests for the Bulks data model.


.. py:class:: TestAdsorbates(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.Adsorbates`\ ]

   Serde tests for the Adsorbates data model.


.. py:class:: TestAtoms(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.Atoms`\ ]

   Serde tests for the Atoms data model.

   .. py:method:: test_to_ase_atoms() -> None



.. py:class:: TestSlabMetadata(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.SlabMetadata`\ ]

   Serde tests for the SlabMetadata data model.


.. py:class:: TestSlab(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.Slab`\ ]

   Serde tests for the Slab data model.


.. py:class:: TestSlabs(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.Slabs`\ ]

   Serde tests for the Slabs data model.


.. py:class:: TestAdsorbateSlabConfigs(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.AdsorbateSlabConfigs`\ ]

   Serde tests for the AdsorbateSlabConfigs data model.


.. py:class:: TestAdsorbateSlabRelaxationsSystem(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationsSystem`\ ]

   Serde tests for the AdsorbateSlabRelaxationsSystem data model.


.. py:class:: TestAdsorbateSlabRelaxationsRequest(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationsRequest`\ ]

   Serde tests for the AdsorbateSlabRelaxationsRequest data model.


.. py:class:: TestAdsorbateSlabRelaxationsRequest_req_fields_only(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationsRequest`\ ]

   Serde tests for the AdsorbateSlabRelaxationsRequest data model in which
   optional fields are omitted.


.. py:class:: TestAdsorbateSlabRelaxationResult(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationResult`\ ]

   Serde tests for the AdsorbateSlabRelaxationResult data model.

   .. py:method:: test_to_ase_atoms() -> None



.. py:class:: TestAdsorbateSlabRelaxationResult_req_fields_only(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationResult`\ ]

   Serde tests for the AdsorbateSlabRelaxationResult data model in which
   optional fields are omitted.


.. py:class:: TestAdsorbateSlabRelaxationsResults(*args: Any, **kwargs: Any)


   Bases: :py:obj:`ModelTestWrapper`\ [\ :py:obj:`fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationsResults`\ ]

   Serde tests for the AdsorbateSlabRelaxationsResults data model.


