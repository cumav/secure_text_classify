## Create a docker image with uwsgi/nginx/tensorflow

This is a edited Dockerfile originally from https://github.com/tiangolo/uwsgi-nginx-docker

Start building the image by running:
``docker build .``

After the image has been build, select the image id and tag it:
``docker tag <image-id> tensorflow/nginx:latest``

