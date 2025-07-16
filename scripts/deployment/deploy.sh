#!/usr/bin/env bash
set -e
docker-compose -f docker-compose.yml -f config/docker/docker-compose.prod.yml up -d --build
