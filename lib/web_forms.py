from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import StringField


class ImageForm(FlaskForm):
    observation_id = StringField("observation_id")
    image = FileField("image", validators=[
        FileRequired(message="Please include 'image' field.")
    ])
    lat = StringField("lat")
    lng = StringField("lng")
    geomodel = StringField("geomodel")
    elevation_model = StringField("elevation_model")
    format = StringField("format")
