try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    config = {
            'description': 'Hybrid Context Aware Recommender',
            'author': 'Laertis Pappas',
            'url': '',
            'download_url': '',
            'author_email': 'laertis.pappas@gmail.com',
            'version': '0.1',
            'install_requires': ['nose'],
            'packages': ['cars'],
            'scripts': [],
            'name': 'projectname'
            }
    setup(**config)
