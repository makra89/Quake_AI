![Logo](resources/logo.png)

# Quake AI - Work in Progress!

I got inspired by the work done in https://medium.com/swlh/training-a-neural-network-to-autoshoot-in-fps-games-e105f27ec1a0 and wanted to try it myself. I think games are a huge playground for computer vision and especially testing deep learning approaches. Why Quake? Simple, you have to start somewhere and you can force enemy models to be green glowing skeletons :)

This project is NOT meant to be used in online play, but should only be used for playing around offline!

# Aimbot

The aimbot is based on:
- Yolov3-tiny neural net for detection (~30 FPS)
- OpenCV MedianFlow tracker (~100 FPS)

It is still an early version of the aimbot with a lot of flaws:
- No tracker "history" --> jumps around a lot if there are multiple enemies present
- Small enemies are hard to detect --> try Yolov3 or Yolov3-tiny-3l

Quake AI comes with a complete framework for capturing training images, annotating the images and training the neural net.
At the moment it is hard-coded for Quake Live, but with some effort it should be possible to port it to other games.

[![Some aimbot action](https://img.youtube.com/vi/ArPtX1xwGiY/hqdefault.jpg)](https://youtu.be/ArPtX1xwGiY)

# Triggerbot

The triggerbot detects enemies in a (default) 160x160 pixel field-of-view and uses an InceptionNet-inspired architecture. It triggers as soon as the aim touches the edges of an enemy. Since it is a very small neural net (~700k parameters) it is really fast. The downside is that there are a lot of false-positives, for example it also triggers when the railgun ammunition box is in the fov. 

The main reason why I started with the triggerbot was the idea that it could help with annotating pictures for the aimbot training, but due to the high rate of false-positives this did not really work out. It is implemented, but I don't recommend using it.

# Prerequisites

Quake AI can only detect green glowing skeletons. 
There are a bunch of settings required in Quake Live for our glorious AI to work:

- Run Quake Live in Windowed Mode (not borderless windowed, i mean windowed!)
- Set the Aim size to the "dot" and the width to "16"
- Force the enemy player model to "bones", choose skin "Bright" and set all enemy colors to green
- Switch mouse input to -1 (windows mouse), direct input does not work for me

# Dependencies

Most imortant packages needed:

- Tensorflow/Keras (I tested with v2.2)
- pyautogui
- mss

# Setup

- Run the script start_quake_ai.py in the main folder
- Either choose an existing config or type in a config name to create a default one
- Open Quake Live and start a match (Instagib works well)
- Push the Triggerbot or Aimbot start button
- Enjoy :)

