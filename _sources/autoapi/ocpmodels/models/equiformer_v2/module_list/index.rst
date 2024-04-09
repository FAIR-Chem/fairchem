:py:mod:`ocpmodels.models.equiformer_v2.module_list`
====================================================

.. py:module:: ocpmodels.models.equiformer_v2.module_list


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.equiformer_v2.module_list.ModuleListInfo




.. py:class:: ModuleListInfo(info_str, modules=None)


   Bases: :py:obj:`torch.nn.ModuleList`

   Holds submodules in a list.

   :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
   modules it contains are properly registered, and will be visible by all
   :class:`~torch.nn.Module` methods.

   :param modules: an iterable of modules to add
   :type modules: iterable, optional

   Example::

       class MyModule(nn.Module):
           def __init__(self):
               super().__init__()
               self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

           def forward(self, x):
               # ModuleList can act as an iterable, or be indexed using ints
               for i, l in enumerate(self.linears):
                   x = self.linears[i // 2](x) + l(x)
               return x

   .. py:method:: __repr__() -> str

      Return a custom repr for ModuleList that compresses repeated module representations.



