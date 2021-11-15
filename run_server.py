from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField
from wtforms.validators import DataRequired
from wtforms.widgets import html5 as h5widgets
import dill
import pandas as pd
from waitress import serve


class ClientDataForm(FlaskForm):
    Pclass = SelectField('Ticket class', choices=[(1, "First class"), (2, "Second Class"), (3, "Third class")], default=1)
    Sex = SelectField('Sex', choices=[(0, "Male"), (1, "Female")])
    Age = IntegerField('Age', validators=[DataRequired()], widget=h5widgets.NumberInput(min=0, max=100, step=5))
    Embarked = SelectField('Port of Embarked', choices=[(1, "Southampton (UK)"), (2, "Cherbourg (France)"), (3, "Queenstown (Ireland)")])


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    #print(model)

modelpath = "models/logreg.dill"
load_model(modelpath)

def get_prediction(Pclass, Sex, Age, Embarked):
    body = {'Pclass': [Pclass], 'Sex': [Sex],'Age': [Age], 'Embarked': [Embarked]}
    preds = model.predict_proba(pd.DataFrame(body))
    return preds

@app.route('/', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    data = dict()
    if request.method == 'POST':
        data['Pclass'] = request.form.get('Pclass')
        data['Sex'] = request.form.get('Sex')
        data['Age'] = request.form.get('Age')
        data['Embarked'] = request.form.get('Embarked')

        response = (get_prediction(data['Pclass'],
                                    data['Sex'],
                                    data['Age'],
                                    data['Embarked']))
        #print(response[0][1])
        return redirect(url_for('predicted', response=response[0][1]))
    return render_template('index.html', form=form)

@app.route('/predicted/<response>')
def predicted(response):
    #print(response)
    return render_template('predicted.html', response=response)


#if __name__ == '__main__':
#app.run(host='0.0.0.0', port=8181, debug=True)
serve(app, host="0.0.0.0", port=8181)
