# 09/26

CGCNN on dropbox.com/s/ckthvf2tzb6nzql/2019_09_23_co_absorption_data.zip.

- 2019-09-26-16-25-14: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 14000`
- 2019-09-26-16-26-33: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 7000`
- 2019-09-26-16-28-02: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 3500`
- 2019-09-26-16-28-58: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 1750`

Changed default optimizer to Adam and (initial) learning rate to 1e-3.

- 2019-09-26-19-10-52: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 14000`

Playing with other hyperparams.

- 2019-10-02-01-22-17: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 14000 --n-conv 8 --n-h 1`
- 2019-10-02-01-26-23: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 14000 --n-conv 8 --n-h 4`

Switching to L1 loss.

- 2019-10-02-01-29-27: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 14000 --n-conv 8 --n-h 1`
- 2019-10-02-01-30-12: `python main.py data/data/2019_09_23/energy-regression --epochs 1000 --train-size 14000 --n-conv 8 --n-h 4`
