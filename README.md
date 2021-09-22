![Logo](resources/logo.png)

# Quake AI

I got inspired by the work done in https://medium.com/swlh/training-a-neural-network-to-autoshoot-in-fps-games-e105f27ec1a0 and wanted to try it myself. I think games are a huge playground for computer vision and especially testing deep learning approaches. Why Quake? Simple, you have to start somewhere and you can force enemy models to be green glowing skeletons :)

This project is NOT meant to be used in online play, but should only be used for playing around offline!

# Aimbot

The aimbot is based on:
- Yolov3-tiny-3l neural net for detection (~30 FPS). "3l" means that it has three detection heads instead of two
- OpenCV MedianFlow tracker (~100 FPS)

Quake AI comes with a complete framework for capturing training images, annotating the images and training the neural net.
At the moment it is hard-coded for Quake Live, but with some effort it should be possible to port it to other games.
The aimbot behavior can be configured using the configuration file, right now it is configured more as an aim assist than an aimbot.

[![Some aimbot action](https://img.youtube.com/vi/ArPtX1xwGiY/hqdefault.jpg)](https://youtu.be/ArPtX1xwGiY)

Yolov3-tiny-3l only would not be enough with its ~ 30 FPS.
The MedianFlow tracker helps a lot in-between to yolo detections.
In the following video i deactivated the mouse movement to show the difference between detection/tracking FPS.

[![Detection/Tracking](https://img.youtube.com/vi/Mi6IjBMavg8/hqdefault.jpg)](https://youtu.be/Mi6IjBMavg8)

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

# Setup

- Run the script start_quake_ai.py in the main folder
- Either choose an existing config or type in a config name to create a default one
- Open Quake Live and start a match (Instagib works well)
- Push the Triggerbot or Aimbot start button
- When using the aimbot activate it pressing "e"
- The aimbot overlay can be disabled via the configuration
- Enjoy :)

