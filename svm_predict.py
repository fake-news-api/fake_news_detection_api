# warnings
import warnings
warnings.filterwarnings('ignore')
import joblib

model_dir = './SVM_models/'

models = {}
names = ['text_', 'title_text_', 'author_text_', 'title_author_text_']

for name in names:
    print("Load:", name)
    loaded_model = joblib.load(model_dir + f"svm_{name}model.pkl")
    models[name] = loaded_model
    print("")


def predict(text, title = "", author = ""):
  try:
    item = ""
    data_var = ""
    
    if title:
      item += title + ". "
      data_var += "title_"

    if author:
      item += author + ". "
      data_var += "author_"

    if text:
      item += text
      data_var += "text_"
    

    model = models[data_var]

    prediction = model.predict([item])[0]
    return {"outcome": ["Real", "Fake"][prediction]}
  except:
    return {"outcome": "error"}
  # return output
