ocpapi.client.client
====================

.. py:module:: ocpapi.client.client


Exceptions
----------

.. autoapisummary::

   ocpapi.client.client.RequestException
   ocpapi.client.client.NonRetryableRequestException
   ocpapi.client.client.RateLimitExceededException


Classes
-------

.. autoapisummary::

   ocpapi.client.client.Client


Module Contents
---------------

.. py:exception:: RequestException(method: str, url: str, cause: str)

   Bases: :py:obj:`Exception`


   Exception raised any time there is an error while making an API call.


.. py:exception:: NonRetryableRequestException(method: str, url: str, cause: str)

   Bases: :py:obj:`RequestException`


   Exception raised when an API call is rejected for a reason that will
   not succeed on retry. For example, this might include a malformed request
   or action that is not allowed.


.. py:exception:: RateLimitExceededException(method: str, url: str, retry_after: datetime.timedelta | None = None)

   Bases: :py:obj:`RequestException`


   Exception raised when an API call is rejected because a rate limit has
   been exceeded.

   .. attribute:: retry_after

      If known, the time to wait before the next attempt to
      call the API should be made.


   .. py:attribute:: retry_after
      :type:  datetime.timedelta | None


.. py:class:: Client(host: str = 'open-catalyst-api.metademolab.com', scheme: str = 'https')

   Exposes each route in the OCP API as a method.


   .. py:attribute:: _host


   .. py:attribute:: _base_url


   .. py:property:: host
      :type: str


      The host being called by this client.


   .. py:method:: get_models() -> ocpapi.client.models.Models
      :async:


      Fetch the list of models that are supported in the API.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The models that are supported in the API.



   .. py:method:: get_bulks() -> ocpapi.client.models.Bulks
      :async:


      Fetch the list of bulk materials that are supported in the API.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The bulks that are supported throughout the API.



   .. py:method:: get_adsorbates() -> ocpapi.client.models.Adsorbates
      :async:


      Fetch the list of adsorbates that are supported in the API.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The adsorbates that are supported throughout the API.



   .. py:method:: get_slabs(bulk: str | ocpapi.client.models.Bulk) -> ocpapi.client.models.Slabs
      :async:


      Get a unique list of slabs for the input bulk structure.

      :param bulk: If a string, the id of the bulk to use. Otherwise the Bulk
                   instance to use.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: Slabs for each of the unique surfaces of the material.



   .. py:method:: get_adsorbate_slab_configs(adsorbate: str, slab: ocpapi.client.models.Slab) -> ocpapi.client.models.AdsorbateSlabConfigs
      :async:


      Get a list of possible binding sites for the input adsorbate on the
      input slab.

      :param adsorbate: Description of the the adsorbate to place.
      :param slab: Information about the slab on which the adsorbate should
                   be placed.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: Configurations for each adsorbate binding site on the slab.



   .. py:method:: submit_adsorbate_slab_relaxations(adsorbate: str, adsorbate_configs: list[ocpapi.client.models.Atoms], bulk: ocpapi.client.models.Bulk, slab: ocpapi.client.models.Slab, model: str, ephemeral: bool = False) -> ocpapi.client.models.AdsorbateSlabRelaxationsSystem
      :async:


      Starts relaxations of the input adsorbate configurations on the input
      slab using energies and forces returned by the input model. Relaxations
      are run asynchronously and results can be fetched using the system id
      that is returned from this method.

      :param adsorbate: Description of the adsorbate being simulated.
      :param adsorbate_configs: List of adsorbate configurations to relax. This
                                should only include the adsorbates themselves; the surface is
                                defined in the "slab" field that is a peer to this one.
      :param bulk: Details of the bulk material being simulated.
      :param slab: The structure of the slab on which adsorbates are placed.
      :param model: The model that will be used to evaluate energies and forces
                    during relaxations.
      :param ephemeral: If False (default), any later attempt to delete the
                        generated relaxations will be rejected. If True, deleting the
                        relaxations will be allowed, which is generally useful for
                        testing when there is no reason for results to be persisted.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: IDs of the relaxations.



   .. py:method:: get_adsorbate_slab_relaxations_request(system_id: str) -> ocpapi.client.models.AdsorbateSlabRelaxationsRequest
      :async:


      Fetches the original relaxations request for the input system.

      :param system_id: The ID of the system to fetch.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The original request that was made when submitting relaxations.



   .. py:method:: get_adsorbate_slab_relaxations_results(system_id: str, config_ids: list[int] | None = None, fields: list[str] | None = None) -> ocpapi.client.models.AdsorbateSlabRelaxationsResults
      :async:


      Fetches relaxation results for the input system.

      :param system_id: The system id of the relaxations.
      :param config_ids: If defined and not empty, a subset of configurations
                         to fetch. Otherwise all configurations are returned.
      :param fields: If defined and not empty, a subset of fields in each
                     configuration to fetch. Otherwise all fields are returned.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The relaxation results for each configuration in the system.



   .. py:method:: delete_adsorbate_slab_relaxations(system_id: str) -> None
      :async:


      Deletes all relaxation results for the input system.

      :param system_id: The ID of the system to delete.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.



   .. py:method:: _run_request(path: str, method: str, **kwargs) -> str
      :async:


      Helper method that runs the input request on a thread so that
      it doesn't block the event loop on the calling thread.

      :param path: The URL path to make the request against.
      :param method: The HTTP method to use (GET, POST, etc.).

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The response body from the request as a string.



