from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_pybo():
    return 'Hello, Pybo!'

@app.route('/classify')
def classification():
    
    return 

if __name__=='__main__':
    app.run(debug=True)