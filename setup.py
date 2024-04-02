from setuptools import find_packages, setup

setup(
    name='few-shot-benchmark',
    packages=find_packages('src'),
    package_dir = { '' : 'src' },
    version='0.1.0',
    description='Classify work experience from Linkedin using a few-shot learning AI model',
    author='Estéban DARTUS, Nino HAMEL, Robin MENEUST, Jérémy SAELEN, Mathis TEMPO',
)