core.models.model_registry
==========================

.. py:module:: core.models.model_registry

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.models.model_registry.MODEL_REGISTRY
   core.models.model_registry.available_pretrained_models


Classes
-------

.. autoapisummary::

   core.models.model_registry.HuggingFaceModel
   core.models.model_registry.URLModel
   core.models.model_registry.ModelRegistry


Functions
---------

.. autoapisummary::

   core.models.model_registry.model_name_to_local_file


Module Contents
---------------

.. py:class:: HuggingFaceModel(/, **data: Any)

   Bases: :py:obj:`pydantic.BaseModel`


   Usage docs: https://docs.pydantic.dev/2.10/concepts/models/

   A base class for creating Pydantic models.

   .. attribute:: __class_vars__

      The names of the class variables defined on the model.

   .. attribute:: __private_attributes__

      Metadata about the private attributes of the model.

   .. attribute:: __signature__

      The synthesized `__init__` [`Signature`][inspect.Signature] of the model.

   .. attribute:: __pydantic_complete__

      Whether model building is completed, or if there are still undefined fields.

   .. attribute:: __pydantic_core_schema__

      The core schema of the model.

   .. attribute:: __pydantic_custom_init__

      Whether the model has a custom `__init__` function.

   .. attribute:: __pydantic_decorators__

      Metadata containing the decorators defined on the model.
      This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.

   .. attribute:: __pydantic_generic_metadata__

      Metadata for generic models; contains data used for a similar purpose to
      __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.

   .. attribute:: __pydantic_parent_namespace__

      Parent namespace of the model, used for automatic rebuilding of models.

   .. attribute:: __pydantic_post_init__

      The name of the post-init method for the model, if defined.

   .. attribute:: __pydantic_root_model__

      Whether the model is a [`RootModel`][pydantic.root_model.RootModel].

   .. attribute:: __pydantic_serializer__

      The `pydantic-core` `SchemaSerializer` used to dump instances of the model.

   .. attribute:: __pydantic_validator__

      The `pydantic-core` `SchemaValidator` used to validate instances of the model.

   .. attribute:: __pydantic_fields__

      A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.

   .. attribute:: __pydantic_computed_fields__

      A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.

   .. attribute:: __pydantic_extra__

      A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra]
      is set to `'allow'`.

   .. attribute:: __pydantic_fields_set__

      The names of fields explicitly set during instantiation.

   .. attribute:: __pydantic_private__

      Values of private attributes set on the model instance.


   .. py:attribute:: type
      :type:  Literal['huggingface_hub']


   .. py:attribute:: repo_id
      :type:  Literal['fairchem/OMAT24']


   .. py:attribute:: filename
      :type:  str


.. py:class:: URLModel(/, **data: Any)

   Bases: :py:obj:`pydantic.BaseModel`


   Usage docs: https://docs.pydantic.dev/2.10/concepts/models/

   A base class for creating Pydantic models.

   .. attribute:: __class_vars__

      The names of the class variables defined on the model.

   .. attribute:: __private_attributes__

      Metadata about the private attributes of the model.

   .. attribute:: __signature__

      The synthesized `__init__` [`Signature`][inspect.Signature] of the model.

   .. attribute:: __pydantic_complete__

      Whether model building is completed, or if there are still undefined fields.

   .. attribute:: __pydantic_core_schema__

      The core schema of the model.

   .. attribute:: __pydantic_custom_init__

      Whether the model has a custom `__init__` function.

   .. attribute:: __pydantic_decorators__

      Metadata containing the decorators defined on the model.
      This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.

   .. attribute:: __pydantic_generic_metadata__

      Metadata for generic models; contains data used for a similar purpose to
      __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.

   .. attribute:: __pydantic_parent_namespace__

      Parent namespace of the model, used for automatic rebuilding of models.

   .. attribute:: __pydantic_post_init__

      The name of the post-init method for the model, if defined.

   .. attribute:: __pydantic_root_model__

      Whether the model is a [`RootModel`][pydantic.root_model.RootModel].

   .. attribute:: __pydantic_serializer__

      The `pydantic-core` `SchemaSerializer` used to dump instances of the model.

   .. attribute:: __pydantic_validator__

      The `pydantic-core` `SchemaValidator` used to validate instances of the model.

   .. attribute:: __pydantic_fields__

      A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.

   .. attribute:: __pydantic_computed_fields__

      A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.

   .. attribute:: __pydantic_extra__

      A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra]
      is set to `'allow'`.

   .. attribute:: __pydantic_fields_set__

      The names of fields explicitly set during instantiation.

   .. attribute:: __pydantic_private__

      Values of private attributes set on the model instance.


   .. py:attribute:: url
      :type:  str


   .. py:attribute:: type
      :type:  Literal['url']


.. py:class:: ModelRegistry(/, **data: Any)

   Bases: :py:obj:`pydantic.BaseModel`


   Usage docs: https://docs.pydantic.dev/2.10/concepts/models/

   A base class for creating Pydantic models.

   .. attribute:: __class_vars__

      The names of the class variables defined on the model.

   .. attribute:: __private_attributes__

      Metadata about the private attributes of the model.

   .. attribute:: __signature__

      The synthesized `__init__` [`Signature`][inspect.Signature] of the model.

   .. attribute:: __pydantic_complete__

      Whether model building is completed, or if there are still undefined fields.

   .. attribute:: __pydantic_core_schema__

      The core schema of the model.

   .. attribute:: __pydantic_custom_init__

      Whether the model has a custom `__init__` function.

   .. attribute:: __pydantic_decorators__

      Metadata containing the decorators defined on the model.
      This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.

   .. attribute:: __pydantic_generic_metadata__

      Metadata for generic models; contains data used for a similar purpose to
      __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.

   .. attribute:: __pydantic_parent_namespace__

      Parent namespace of the model, used for automatic rebuilding of models.

   .. attribute:: __pydantic_post_init__

      The name of the post-init method for the model, if defined.

   .. attribute:: __pydantic_root_model__

      Whether the model is a [`RootModel`][pydantic.root_model.RootModel].

   .. attribute:: __pydantic_serializer__

      The `pydantic-core` `SchemaSerializer` used to dump instances of the model.

   .. attribute:: __pydantic_validator__

      The `pydantic-core` `SchemaValidator` used to validate instances of the model.

   .. attribute:: __pydantic_fields__

      A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.

   .. attribute:: __pydantic_computed_fields__

      A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.

   .. attribute:: __pydantic_extra__

      A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra]
      is set to `'allow'`.

   .. attribute:: __pydantic_fields_set__

      The names of fields explicitly set during instantiation.

   .. attribute:: __pydantic_private__

      Values of private attributes set on the model instance.


   .. py:attribute:: models
      :type:  dict[str, pydantic.AnyUrl | HuggingFaceModel | URLModel]


.. py:data:: MODEL_REGISTRY

.. py:data:: available_pretrained_models

.. py:function:: model_name_to_local_file(model_name: str, local_cache: str | pathlib.Path) -> str

   Download a pretrained checkpoint if it does not exist already

   :param model_name: the model name. See available_pretrained_checkpoints.
   :type model_name: str
   :param local_cache: path to local cache directory
   :type local_cache: str or Path

   :returns: local path to checkpoint file
   :rtype: str


