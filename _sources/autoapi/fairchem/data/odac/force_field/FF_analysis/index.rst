:py:mod:`fairchem.data.odac.force_field.FF_analysis`
====================================================

.. py:module:: fairchem.data.odac.force_field.FF_analysis


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.data.odac.force_field.FF_analysis.get_data
   fairchem.data.odac.force_field.FF_analysis.binned_average
   fairchem.data.odac.force_field.FF_analysis.bin_plot
   fairchem.data.odac.force_field.FF_analysis.get_Fig4a
   fairchem.data.odac.force_field.FF_analysis.get_Fig4b
   fairchem.data.odac.force_field.FF_analysis.get_Fig4c
   fairchem.data.odac.force_field.FF_analysis.get_Fig4d
   fairchem.data.odac.force_field.FF_analysis.phys_err
   fairchem.data.odac.force_field.FF_analysis.chem_err



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.data.odac.force_field.FF_analysis.infile


.. py:function:: get_data(infile, limit=2)


.. py:function:: binned_average(DFT_ads, pred_err, bins)


.. py:function:: bin_plot(ax, bins, heights, **kwargs)


.. py:function:: get_Fig4a(raw_error_CO2, raw_error_H2O, b=20, outfile='Fig5a.png')


.. py:function:: get_Fig4b(int_DFT_CO2, err_CO2, int_DFT_H2O, err_H2O, outfile='Fig5b.png')


.. py:function:: get_Fig4c(DFT_CO2, err_CO2, outfile='Fig5c.png')


.. py:function:: get_Fig4d(DFT_H2O, err_H2O, outfile='Fig5d.png')


.. py:function:: phys_err(DFT, FF)


.. py:function:: chem_err(DFT, FF)


.. py:data:: infile
   :value: '/storage/home/hcoda1/8/lbrabson3/p-amedford6-0/s2ef/final/data_w_oms.json'

   

