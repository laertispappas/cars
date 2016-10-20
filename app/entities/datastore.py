from sqlalchemy import ForeignKey
from sqlalchemy import Table, MetaData, create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql.sqltypes import DATETIME

Session = sessionmaker()
Base = declarative_base()

# Responsible to connect to PostgreSQL in order to extract data
# for our algorithms
class Datastore(object):
    def __init__(self):
        self.con = self.connect('laertispappas', 'pappas', 'my_movies_development')
        Session.configure(bind=self.con)
        self.session = Session()

    def connect(user, password, db, host='localhost', port=5432):
        '''Returns a connection and a metadata object'''
        # We connect with the help of the PostgreSQL URL
        # postgresql://federer:grandestslam@localhost:5432/tennis
        url = 'postgresql://{}:{}@{}:{}/{}'
        url = url.format(user, password, host, port, db)

        # The return value of create_engine() is our connection object
        # con = create_engine(url, client_encoding='utf8')
        con = create_engine('postgresql://laertispappas:pappas@localhost:5432/my_movies_development', client_encoding='utf8')

        return con

    def users(self):
        return self.session.query(User).all()

    def total_users(self):
        return self.session.query(User.id).count()

    def movies(self):
        return self.session.query(Movie).all()

    def total_movies(self):
        return self.session.query(Movie.id).count()

    def ratings(self):
        return self.session.query(Rating).all()

    def total_ratings(self):
        return self.session.query(Rating.id).count()

    def rating_conditions(self):
        return self.session.query(RatingCondition).all()

    def conditions(self):
        return self.session.query(Condition).all()

    def contexts(self):
        return self.session.query(Context).all()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer)
    email = Column(String)
    gender = Column(String)
    birthday = Column(DATETIME)

    # TODO: User city
    ratings = relationship("Rating", back_populates="user")

    def __repr__(self):
        return "<User(email='%s')>" % (self.email)


movie_actors = Table('movie_actors', Base.metadata,
                     Column('actor_id', ForeignKey('actors.id'), primary_key=True),
                     Column('movie_id', ForeignKey('movies.id'), primary_key=True))

movie_genres = Table('movie_genres', Base.metadata,
                     Column('movie_id', ForeignKey('movies.id'), primary_key=True),
                     Column('genre_id', ForeignKey('genres.id'), primary_key=True))
class Genre(Base):
    __tablename__ = 'genres'
    id = Column(Integer, primary_key=True)
    name = Column(String)

    movies = relationship('Movie', secondary=movie_genres, back_populates='genres')

    def __repr__(self):
        return "<Genre(name='%s')>" % (self.name)

class Actor(Base):
    __tablename__ = 'actors'
    id = Column(Integer, primary_key=True)
    name = Column(String)

    movies = relationship('Movie', secondary=movie_actors, back_populates='actors')

    def __repr__(self):
        return "<Actor(name='%s')>" % (self.name.encode('utf-8'))


class Movie(Base):
    __tablename__ = 'movies'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    director = Column(String)
    language = Column(String)
    country = Column(String)
    year = Column(String)
    budget = Column(String)

    ratings = relationship("Rating", back_populates="movie")
    actors = relationship("Actor", secondary=movie_actors, back_populates="movies")
    genres = relationship('Genre', secondary=movie_genres, back_populates='movies')

    def __repr__(self):
        return "<Movie(title='%s')>" % (self.title)

class Rating(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    movie_id = Column(Integer, ForeignKey('movies.id'))
    score = Column(Integer)

    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")
    rating_conditions = relationship("RatingCondition", back_populates="rating")

    def __repr__(self):
        return "<Rating(user='%s', movie='%s', score='%s', rating_conditions='%s')>" % (self.user, self.movie, self.score, self.rating_conditions)


class RatingCondition(Base):
    __tablename__ = 'rating_conditions'
    id = Column(Integer, primary_key=True)
    rating_id = Column(Integer, ForeignKey('ratings.id'))
    condition_id = Column(Integer, ForeignKey('conditions.id'))

    rating = relationship("Rating", back_populates="rating_conditions")
    condition = relationship("Condition", back_populates="rating_conditions")

    def __repr__(self):
        return "<RateCondition(Context='%s', Condition='%s')>" % (self.condition.context.name, self.condition.name)

class Condition(Base):
    __tablename__ = 'conditions'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    context_id = Column(Integer, ForeignKey('contexts.id'))

    context = relationship("Context", back_populates="conditions")
    rating_conditions = relationship("RatingCondition", back_populates="condition")

    def __repr__(self):
        return "<Condition(Context='%s', name='%s')>" % (self.context, self.name)


class Context(Base):
    __tablename__ = 'contexts'
    id = Column(Integer, primary_key=True)
    name = Column(String)

    conditions = relationship("Condition", back_populates="context")

    def __repr__(self):
        return "<Context(name='%s')>" % (self.name)

