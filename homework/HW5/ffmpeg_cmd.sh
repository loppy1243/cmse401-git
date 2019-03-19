#!/bin/bash

exec ffmpeg -r $1 -i images/file%05d.png -c:v libx264 -vf fps=24 -pix_fmt yuv420p out.mp4
