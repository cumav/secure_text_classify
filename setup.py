from setuptools import setup

with open("README.md") as readme:
    long_description = readme.read()

setup(
    name='secure_anit_spam',
    version='0.0.1',
    description='Language processing api tools',
    long_description=long_description,
    license='proprietary software, no public license',
    author='Marvin Gurka',
    author_email='genmethat@gmail.com',
    url='https://genmethat.com',
    packages=['secure_anit_spam'],
    install_requires=[
        'tensorflow_hub>=0.6.0',
        'tensorflow>=2.0.0',
        'tensorflow_text',
        'pandas',
        'keras'
    ]
)