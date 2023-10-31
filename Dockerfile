FROM python:3.11.6

RUN apt-get update && apt-get install -y libgdal-dev

RUN useradd -ms /bin/bash inaturalist
USER inaturalist

ENV PATH="/home/inaturalist/.local/bin:${PATH}"

RUN pip install --upgrade pip

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY --chown=inaturalist:inaturalist ./requirements.txt /code/requirements.txt

# install dependencies
RUN pip install -r requirements.txt

COPY --chown=inaturalist:inaturalist . /code

# command to run on container start
CMD [ "python", "app.py" ]
