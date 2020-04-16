#!/bin/sh

mkdir -p data
cd data

# CO adsorption experiments from Zack / Jun.
# Preprocessed into data format pytorch-geometric likes.
wget https://www.dropbox.com/s/ee0hfoc480zzlgb/ulissigroup_co.zip
unzip ulissigroup_co.zip
rm ulissigroup_co.zip

# CO, H, N, O, and OH adsorption on various slabs from Kevin.
# Raw ase db files.
wget https://www.dropbox.com/s/c01e1b2cpr8wsu4/gasdb.zip
unzip gasdb.zip
rm gasdb.zip
