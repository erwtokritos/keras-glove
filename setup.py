from setuptools import setup

with open('requirements.txt', 'r') as fp:
    requirements = [x.strip() for x in fp.readlines()]

setup(name='keras-glove',
      version='0.1.0',
      description='GloVe implementation in Keras',
      author='Thanos Papaoikonomou',
      author_email='thanos.papaoikonomou@gmail.com',
      install_requires=requirements,
      entry_points={
          'console_scripts': [
              'kglove = orchestrator:cli',
          ]
      }
      )
