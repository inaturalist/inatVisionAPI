FROM python:3.10.5

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
