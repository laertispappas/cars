import os, sys
sys.path.insert(0, os.path.abspath(".."))

from app.dataset.data_object import DataObject
from app.dataset.loader import Loader


def main():
    data_object = DataObject()
    data_object.print_specs()
    print data_object.ratings.head
if __name__ == "__main__": main()