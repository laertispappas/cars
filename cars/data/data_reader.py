import xlrd
from data_object import DataObject

class DataReader(object):
  RATING_COLUMNS = {'user_id': 0, 'movie_id': 1, 'rating': 2, 
        'time': 7, 'daytype': 8, 'season': 9, 'location': 10,
        'weather': 11, 'social': 12, 'end_emotion': 13, 
        'dominant_emotion': 14, 'mood': 15, 'physical':16,
        'decision': 17, 'interaction': 8}
  USER_COLUMNS = {'id': 0, 'age': 3, 'gender': 4, 'city': 5, 'country': 6, }
  MOVIE_COLUMNS = {'id': 1, 'director': 19, 'country': 20, 'language': 21, 'year': 22, 'genre':23, 'budget': 27}
 
  def __init__(self, file_path="../dataset/ldos/LDOS-CoMoDa.xls"):
    self.dataset_path = file_path
    self.work_book = xlrd.open_workbook(filename=self.dataset_path) 

  def load(self):  
    sheet = self.work_book.sheet_by_name('Sheet1')
    user_data = []
    movie_data = []
    rating_data = []

    for row in range(1, sheet.nrows):
      rating_data.append(self.__read_xls_columns(sheet, row, RATING_COLUMNS))
      user_data.append(self.__read_xls_columns(sheet, row, USER_COLUMNS))
      movie_data.append(self.__read_xls_columns(sheet, row, MOVIE_COLUMNS))

  def persist_to_db(self):
    raise NotImplemented

  def __read_xls_columns(self, sheet, row, columns_names):
      fields = {col_name: sheet.cell_value(row, columns_names[col_name]) for col_name in columns_names}
      return fields
