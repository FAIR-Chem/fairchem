from setuptools import setup

setup(name='ocdata',
      version='0.0.1',
      description='Module for generating random catalyst adsorption configurations',
      url='http://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset',
      author='Pari Palizhati, Kevin Tran, Javi Heras Domingo, Zack Ulissi, and others',
      author_email='zulissi@andrew.cmu.edu',
      license='GPL',
      platforms=[],
      packages=['ocdata'],
      scripts=[],
      include_package_data=True,
      install_requires=['pymatgen==2020.4.2', 'ase>=3.19.1', 'catkit @ git+ssh://git@github.com:SUNCAT-Center/CatKit.git#egg=catkit'],
      long_description='''Module for generating random catalyst adsorption configurations for high-throughput dataset generation.''',)
