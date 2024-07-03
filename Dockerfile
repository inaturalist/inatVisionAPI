FROM python:3.11.6 as base

RUN apt-get update && apt-get install -y libgdal-dev uwsgi-plugin-python3

RUN useradd -ms /bin/bash inaturalist
USER inaturalist

ENV PATH="/home/inaturalist/.local/bin:${PATH}"

RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /home/inaturalist/vision

# Copy the dependencies file to the working directory
COPY --chown=inaturalist:inaturalist ./requirements.txt /home/inaturalist/vision/requirements.txt

# Install dependencies
RUN UWSGI_EMBED_PLUGINS=stats_pusher_statsd pip install -r requirements.txt

# Copy app and libs
COPY --chown=inaturalist:inaturalist app.py /home/inaturalist/vision
COPY --chown=inaturalist:inaturalist lib /home/inaturalist/vision/lib

# Create directories for the log and static content
RUN mkdir /home/inaturalist/vision/log
RUN mkdir /home/inaturalist/vision/static

# Development target
FROM base AS development

# Run with built-in Flask server
CMD [ "python", "app.py" ]

# Production target with uwsgi
FROM development AS production

# Configure uwsgi
ENV UWSGI_PLUGIN_DIR /usr/lib/uwsgi/plugins
RUN mkdir /home/inaturalist/vision/uwsgi
COPY docker/uwsgi.ini /home/inaturalist/vision/uwsgi.ini

ARG GIT_BRANCH
ARG GIT_COMMIT
ARG IMAGE_TAG
ARG BUILD_DATE

ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_COMMIT=${GIT_COMMIT}
ENV IMAGE_TAG=${IMAGE_TAG}
ENV BUILD_DATE=${BUILD_DATE}

RUN echo "GIT_BRANCH=${GIT_BRANCH}" > /home/inaturalist/vision/build_info
RUN echo "GIT_COMMIT=${GIT_COMMIT}" >> /home/inaturalist/vision/build_info
RUN echo "IMAGE_TAG=${IMAGE_TAG}" >> /home/inaturalist/vision/build_info
RUN echo "BUILD_DATE=${BUILD_DATE}" >> /home/inaturalist/vision/build_info

# Run with uwsgi
CMD ["uwsgi", "--ini", "/home/inaturalist/vision/uwsgi.ini", "--stats", ":1717", "--stats-http"]
