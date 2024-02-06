#!/bin/sh
find /var/vision/static -type f -mmin +10 -exec rm {} \;