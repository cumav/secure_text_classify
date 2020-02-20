#! /usr/bin/env bash

docker build -t tensorflow/nginx:latest docker-tensorflow/.
docker build -t securetext/nginx:latest .
docker container run --publish 80:80 --detach securetext/nginx:latest
