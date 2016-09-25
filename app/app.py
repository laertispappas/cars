import os, sys
sys.path.insert(0, os.path.abspath(".."))

from app.dataset.loader import Loader

def main():
    ratings_data = Loader().load_ratings()
    user_data = Loader().load_user_data()
    movie_data = Loader().load_movie_data()
    print movie_data
    # print ratings_data.values()[2]
    # print user_data
if __name__ == "__main__": main()