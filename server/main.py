from flask import Flask, jsonify
import json
import requests

app = Flask(__name__)


@app.route("/data", methods=['POST', 'GET'])
def data():
    #dictToReturn = {'answer':42}
    #return jsonify(dictToReturn)
    return "Hello World!"

if __name__ == '__main__':
    app.run()
# $ flask run --host=0.0.0.0
# flask --app main run --host=0.0.0.0
# POST http://localhost:5000/data
# GET http://localhost:5000/data
# POST http://128.232.250.181:5000/data