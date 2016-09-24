import os, sys
sys.path.insert(0, os.path.abspath(".."))

from app.entities.datastore import Datastore, User


def main():
    store = Datastore()
    # user = store.session.query(User).filter_by(name='ed').first()
    # user = store.session.query(User).first()
    # for email, gender, birthday in store.session.query(User.email, User.gender, User.birthday):
    #     print(email, gender, birthday)
    # for user in store.users():
    #     print(user.email, user.gender, user.birthday, user.ratings)
    # for rating in store.ratings():
    #     print(rating)
    # for rating_condition in store.rating_conditions():
    #     print rating_condition
    # for condition in store.conditions():
    #     print condition
    # for context in store.contexts():
    #     print context
    for movie in store.movies():
        print movie, movie.actors, movie.genres

if __name__ == "__main__": main()