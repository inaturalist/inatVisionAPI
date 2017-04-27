from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired

class ImageForm(FlaskForm):
    image = FileField('image',
        validators=[
            FileRequired(message="Please include 'image' field.")
        ])
