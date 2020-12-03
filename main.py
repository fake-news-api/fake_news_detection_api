from flask import Flask, request
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with
from flask_sqlalchemy import SQLAlchemy
# from newspaper import Article
# import nltk
# nltk.download('punkt')

# from bert_predict import predict
from svm_predict import predict

app = Flask(__name__)
api = Api(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
# db = SQLAlchemy(app)
# print("line12")
# class News_model(db.Model):
#     News_id = db.Column(db.Integer, primary_key = True)
#     News_url = db.Column(db.String, nullable = False)
#     News_out = db.Column(db.String, nullable = False)

#     def __repr__(self):
#         return f"News(News_id={self.News_id}, News_url={self.News_url}, News_out={self.News_out})" 

# db.create_all() # run_once

print("line22")
News_put_args = reqparse.RequestParser()
News_put_args.add_argument("text", type = str, help = "text needed", required = True)
News_put_args.add_argument("author", type = str, help = "author needed")
News_put_args.add_argument("title", type = str, help = "title needed")

# app
print("line38")
class FakeNewsDetect(Resource):
    @app.route('/')
    def index():
        return "<h1>Welcome to our server !!</h1>"

    def put(self, running):
        if running != "running":
            return {"outcome": "method not exist"}, 405

        args = News_put_args.parse_args()
        result = predict(text = args["text"], author = args["author"], title = args['title'])
        return result, 201


api.add_resource(FakeNewsDetect, "/<string:running>")
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=True)