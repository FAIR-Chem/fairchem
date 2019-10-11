#!/bin/sh

mkdir -p data

# CO adsorption experiments from Zack / Jun.
wget https://www.dropbox.com/s/ee0hfoc480zzlgb/ulissigroup_co.zip
unzip ulissigroup_co.zip
rm ulissigroup_co.zip

# QM9
wget https://www.dropbox.com/s/vneosc922p8bbn8/qm9.zip
unzip qm9.zip
rm qm9.zip

# Material Project data from Xie & Grossman (2018)
wget https://www.dropbox.com/s/41s1wnmje3krsh6/xie_grossman_mat_proj.zip
unzip xie_grossman_mat_proj.zip
rm xie_grossman_mat_proj.zip
