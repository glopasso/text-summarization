from flask_wtf import FlaskForm
from wtforms import SelectField, FloatField, SubmitField, TextAreaField, StringField
from wtforms.validators import DataRequired
import wtforms.widgets.core

class TextArea(wtforms.widgets.core.TextArea):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, field, **kwargs):
        for arg in self.kwargs:
            if arg not in kwargs:
                kwargs[arg] = self.kwargs[arg]
        return super(TextArea, self).__call__(field, **kwargs)


class SummarizationForm(FlaskForm):
    n_gram = SelectField(u'n-gram', choices=[('1-gram', '1-gram'), ('2-gram', '2-gram'), ('3-gram', '3-gram')])
    tokenizer = SelectField(u'token type', choices=[('stem', 'stem'), ('lemma', 'lemma')])
    threshold_factor = FloatField('Threshold Factor', validators=[DataRequired()])
    text_to_summarize = TextAreaField('Text to summarize', widget=TextArea(rows=10,cols=100))
    submit = SubmitField('Submit')
    prediction = StringField('Prediction', widget=TextArea(rows = 10, cols = 100))
