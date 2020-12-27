#!/bin/bash
set -e

if [ "$1" = 'start' ]; then

    exec daphne -b 0.0.0.0 -p 8000 classifier.asgi:application

elif [ "$1" = 'migrate' ]; then

    python manage.py makemigrations
    python manage.py migrate

    exec daphne -b 0.0.0.0 -p 8000 classifier.asgi:application

fi

exec "$@"