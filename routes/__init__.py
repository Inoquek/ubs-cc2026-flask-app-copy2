from flask import Flask

app = Flask(__name__)
import routes.square
import routes.investigate 
import routes.trivia
import routes.diaries
import routes.blankety
