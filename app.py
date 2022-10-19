from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

app=Flask(__name__)

tfidf=pickle.load(open('vectorizer.pkl', 'rb'))
mnb_model=pickle.load(open('mnb_model.pkl', 'rb'))

def filter_text(text):


    text = text.lower()

    text = nltk.word_tokenize(text)

    result = []
    for char in text:
        if char.isalnum():
            result.append(char)

    text = result[:]
    result.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            result.append(word)

    text = result[:]
    result.clear()

    for item in text:
        result.append(ps.stem(item))

    return " ".join(result)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/check')
def detect():
    return render_template("detector.html")

@app.route('/final_result', methods=['POST'])
def final_result():
    user_input = request.form.get('entry')

    filtered_matter=filter_text(user_input)
    list_input=tfidf.transform([filtered_matter])

    output=mnb_model.predict(list_input)[0]
    print(output)
    answer_list=[]

    if output==0:
        answer=str("Not Spam!")
    else:
        answer=str("!!!SPAM!!!")
    print(answer)

    answer_list.append(answer)

    return render_template("detector.html", data=answer_list)


if __name__=="__main__":
    app.run(debug=True)
