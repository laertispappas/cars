import os, sys
sys.path.insert(0, os.path.abspath(".."))

from app.dataset.loader import Loader

def main():
    ratings = Loader().load_ratings()
    print ratings.values()[2]
    # for movie_title, movie_data in ratings.items():
    #     print movie_data
if __name__ == "__main__": main()