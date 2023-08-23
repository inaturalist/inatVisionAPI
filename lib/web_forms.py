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
    taxon_id = StringField("taxon_id")
    geomodel = StringField("geomodel")
    aggregated = StringField("aggregated")
    score_without_geo = StringField("score_without_geo")
    format = StringField("format")
