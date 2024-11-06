core.modules.loss
=================

.. py:module:: core.modules.loss


Classes
-------

.. autoapisummary::

   core.modules.loss.MAELoss
   core.modules.loss.MSELoss
   core.modules.loss.PerAtomMAELoss
   core.modules.loss.L2NormLoss
   core.modules.loss.DDPLoss


Module Contents
---------------

.. py:class:: MAELoss

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: loss


   .. py:method:: forward(pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor) -> torch.Tensor


.. py:class:: MSELoss

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: loss


   .. py:method:: forward(pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor) -> torch.Tensor


.. py:class:: PerAtomMAELoss

   Bases: :py:obj:`torch.nn.Module`


   Simply divide a loss by the number of atoms/nodes in the graph.
   Current this loss is intened to used with scalar values, not vectors or higher tensors.


   .. py:attribute:: loss


   .. py:method:: forward(pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor) -> torch.Tensor


.. py:class:: L2NormLoss

   Bases: :py:obj:`torch.nn.Module`


   Currently this loss is intened to used with vectors.


   .. py:method:: forward(pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor) -> torch.Tensor


.. py:class:: DDPLoss(loss_name, reduction: Literal['mean', 'sum'])

   Bases: :py:obj:`torch.nn.Module`


   This class is a wrapper around a loss function that does a few things
   like handle nans and importantly ensures the reduction is done
   correctly for DDP. The main issue is that DDP averages gradients
   over replicas — this only works out of the box if the dimension
   you are averaging over is completely consistent across all replicas.
   In our case, that is not true for the number of atoms per batch and
   there are edge cases when the batch size differs between replicas
   e.g. if the dataset size is not divisible by the batch_size.

   Scalars are relatively straightforward to handle, but vectors and higher tensors
   are a bit trickier. Below are two examples of forces.

   Forces input: [Nx3] target: [Nx3]
   Forces are a vector of length 3 (x,y,z) for each atom.
   Number of atoms per batch (N) is different for each DDP replica.

   MSE example:
   #### Local loss computation ####
   local_loss = MSELoss(input, target) -> [Nx3]
   num_samples = local_loss.numel() -> [Nx3]
   local_loss = sum(local_loss [Nx3]) -> [1] sum reduces the loss to a scalar
   global_samples = all_reduce(num_samples) -> [N0x3 + N1x3 + N2x3 + ...] = [1] where N0 is the number of atoms on replica 0
   local_loss = local_loss * world_size / global_samples -> [1]
   #### Global loss computation ####
   global_loss = sum(local_loss / world_size) -> [1]
   == sum(local_loss / global_samples) # this is the desired corrected mean

   Norm example:
   #### Local loss computation ####
   local_loss = L2MAELoss(input, target) -> [N]
   num_samples = local_loss.numel() -> [N]
   local_loss = sum(local_loss [N]) -> [1] sum reduces the loss to a scalar
   global_samples = all_reduce(num_samples) -> [N0 + N1 + N2 + ...] = [1] where N0 is the number of atoms on replica 0
   local_loss = local_loss * world_size / global_samples -> [1]
   #### Global loss computation ####
   global_loss = sum(local_loss / world_size) -> [1]
   == sum(local_loss / global_samples) # this is the desired corrected mean


   .. py:attribute:: loss_fn


   .. py:attribute:: reduction


   .. py:attribute:: reduction_map


   .. py:method:: sum(input, loss, natoms)


   .. py:method:: _ddp_mean(num_samples, loss)


   .. py:method:: mean(input, loss, natoms)


   .. py:method:: _reduction(input, loss, natoms)


   .. py:method:: forward(input: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor)


