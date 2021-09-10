![Logo](resources/logo.png)

# Quake AI - Work in Progress!

I got inspired by the work done in https://medium.com/swlh/training-a-neural-network-to-autoshoot-in-fps-games-e105f27ec1a0 and wanted to try it myself. I think games are a huge playground for computer vision and especially testing deep learning approaches. Why Quake? Simple, you have to start somewhere and you can force enemy models to be green glowing skeletons :)

This project is NOT meant to be used in online play, but should only be used for playing around offline!

# Triggerbot

So far the only thing that is working is the triggerbot. The idea is that the triggerbot, once trained, will do the annotation of enemy bounding boxes for me. At least that is the plan. 

- Fully functional triggerbot (default model + config provided)
- Training environment for triggerbot


[![Some trigger action](https://img.youtube.com/vi/4247DVzgEsU/hqdefault.jpg)](https://youtu.be/4247DVzgEsU)

# Prerequisites

The triggerbot "AI" so far is a really dumb one. It can only detect green glowing skeletons. 
There are a bunch of settings required in Quake Live for our glorious AI to work:

- Run Quake Live in Windowed Mode (not borderless windowed, i mean windowed!)
- Set the Aim size to the "dot" and the width to "16"
- Force the enemy player model to "bones", choose skin "Bright" and set all enemy colors to green

# Dependencies

Most imortant packages needed:

- Tensorflow/Keras (I tested with v2.2)
- pyautogui
- mss

# Setup

- Run the script start_quake_ai.py in the main folder
- Either choose an existing config or type in a config name to create a default one
- Open Quake Live and start a match (Instagib works well)
- Push the Triggerbot start button
- Enjoy :)

