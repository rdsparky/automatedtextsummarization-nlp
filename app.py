from flask import Flask, render_template,request
import mod as s
import nltk
nltk.download('punkt')



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


    if request.method == 'POST':
        data = request.form['content']

      
    
        l , t , im =(s.summary(data))
               
        
    return render_template('result.html', lsap = "\n" + l ,trp = "\n" + t , imur =im)


 

if __name__ == '__main__':
    app.run(debug=True)

