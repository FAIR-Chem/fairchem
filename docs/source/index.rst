Open Catalyst Project
=====================

The Open Catalyst Project is a collaborative research effort between Facebook AI
Research (FAIR) and Carnegie Mellon University’s (CMU) Department of Chemical Engineering.
The aim is to use AI to model and discover new catalysts for use in renewable energy
storage to help in addressing climate change.

Scalable and cost-effective solutions to renewable energy storage are essential to
addressing the world’s rising energy needs while reducing climate change.  As we
increase our reliance on renewable energy sources such as wind and solar, which produce
intermittent power, storage is needed to transfer power from times of peak generation to
peak demand. This may require the storage of power for hours, days, or months. One solution
that offers the potential of scaling to nation-sized grids is the conversion of
renewable energy to other fuels, such as hydrogen. To be widely adopted, this
process requires cost-effective solutions to running chemical reactions.

An open challenge is finding low-cost catalysts to drive these reactions at high rates.
Through the use of quantum mechanical simulations (density functional theory), new
catalyst structures can be tested and evaluated. Unfortunately, the high computational
cost of these simulations limits the number of structures that may be tested. The use of
AI or machine learning may provide a method to efficiently approximate these calculations,
leading to new approaches in finding effective catalysts.

To enable the broader research community to participate in this important project,
we provide baseline models and code at
`Github page <https://github.com/Open-Catalyst-Project/baselines>`_.


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/getting_started
   tutorials/data_playground.ipynb
   tutorials/train_s2ef_example.ipynb
   tutorials/training
   tutorials/submission

..
    .. toctree::
       :maxdepth: 1
       :caption: Modules

       modules/model
       modules/dataset
       modules/trainer

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
