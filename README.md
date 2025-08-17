# Finding Restaurant Success in Singapore using Retrieval-Augmented Generation and Large Language Model SWOT Analysis 

## Overview

This model allows you to find three ideal Singaporean locations given your restaurant's characteristics. It will then run a SWOT analysis on each location with a predicted success score. You can also input your own location with your restaurant's characterisitics to get a SWOT analysis and success score. You may find this project helpful for the following purposes:

* You are looking for ideal locations to open a restaurant in Singapore
* You want a business consulatation regarding the potential success of your restaurant in Singapore
* You want a business consultation regarding your restaurant's current performance in Singapore
* Combining FAISS, RAG, and LLM for urban planning/business applications

## Watch this video for how to use:

## How to use in text if you don't want to watch the video:

1. Clone this project 
2. Set up your OpenAI API key in a new .env file and set OPENAI_API_KEY = "your api key"
3. Depending on whether you want to run a SWOT analysis on  your proposed location, you should uncomment and comment out lines that say to be uncommented and comment out
4. Run main file

## Tweak this project for your own uses

Feel free to clone and use this project for you own purposes. You can webscrape Google Review Data on different cities or countries and replace them into the Data folder if you want to use this model for other locations. You can also tweak the templates in rag_model.py for specifications or whatever you're looking for the LLM to generate. 

## Find a bug?

If you find a bug or want to submit an improvement, please submit an issues using the issues tab above or email me at enyuzhu@mit.edu!

## Please note that this project is still in progress so more training for a more accurate residual adjustment is needed.

