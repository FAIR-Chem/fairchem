:py:mod:`FF_analysis`
=====================

.. py:module:: FF_analysis


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   FF_analysis.get_data
   FF_analysis.binned_average
   FF_analysis.bin_plot
   FF_analysis.get_Fig4a
   FF_analysis.get_Fig4b
   FF_analysis.get_Fig4c
   FF_analysis.get_Fig4d
   FF_analysis.phys_err
   FF_analysis.chem_err



Attributes
~~~~~~~~~~

.. autoapisummary::

   FF_analysis.infile


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

   

