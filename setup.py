from distutils.core import setup

setup(
    name='PyTextDocumentClassification',
    version='0.1',
    packages=['pytdc'],
    url='https://github.com/frugs/PyDeepLearning',
    license='MIT',
    author='Hugo Wainwright',
    author_email='frugs@github.com',
    description='Text document classification library using PyDeepLearning and NumPy.',
    data_files=[("", ["LICENSE.txt"])],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3']
)