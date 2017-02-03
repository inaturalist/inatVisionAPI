from flask_wtf import Form
from flask_wtf.file import FileField, FileRequired

class ImageForm(Form):
    image = FileField('image',
        validators=[
            FileRequired(message="Please include 'image' field.")
        ])
