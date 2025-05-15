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
    return_embedding = StringField("return_embedding")
    common_ancestor_rank_type = StringField("common_ancestor_rank_type")
    human_exclusion = StringField("human_exclusion")
    format = StringField("format")
