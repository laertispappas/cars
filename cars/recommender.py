
class Recommender(object):
  def __init__(self):
    print("Init Recommender")

  def execute(self, debug=True):
    if(debug == True):
      init_model()
      build_model()
      #save_model()
    else:
      load_model()

