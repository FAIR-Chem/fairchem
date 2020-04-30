from setuptools import setup, find_packages
from distutils.command.install import INSTALL_SCHEMES


for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

setup(name='ocdata',
      version='0.0.1',
      description='Module for generating random catalyst adsorption configurations',
      url='http://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset',
      author='Pari Palizhati, Kevin Tran, Javi Heras Domingo, Zack Ulissi, Siddharth Goyal',
      author_email='zulissi@andrew.cmu.edu',
      license='GPL',
      platforms=[],
      packages=find_packages(),
      scripts=[],
      data_files=[('ocdata/base_atoms/ase_dbs', ['ocdata/base_atoms/ase_dbs/bulks.db',
                                                 'ocdata/base_atoms/ase_dbs/adsorbates.db']),
                  ('ocdata/base_atoms/pkls', ['ocdata/base_atoms/pkls/bulks.pkl',
                                              'ocdata/base_atoms/pkls/adsorbates.pkl'])],
      include_package_data=True,
      install_requires=['pymatgen==2020.4.2', 'ase>=3.19.1', 'catkit @ git+https://github.com/SUNCAT-Center/CatKit.git#egg=catkit'],
      long_description='''Module for generating random catalyst adsorption configurations for high-throughput dataset generation.''',)
