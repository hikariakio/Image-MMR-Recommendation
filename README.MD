# Image Recommendation

## Description

Using the Maximal Marginal Relevance (MMR) algorithm, this project ranks images by semantic content, drawing from textual annotations. It features an experimental comparison between Word2Vec and GloVe embeddings to optimize recommendation quality. 

## Installation

```sh
git clone https://github.com/hikariakio/Image-MMR-Recommendation.git
```

## Backend Setup
1. Direct to PyBackend.

2. Create and go inside to the virtual environment (python 3.11.6)

3. Install the packages from requirements.txt

4. Download Precomputed Caches

```
https://s3.ap-southeast-2.amazonaws.com/yginnovatory.com/MMR_Recommendation/Sim_matrix_glove.csv

https://s3.ap-southeast-2.amazonaws.com/yginnovatory.com/MMR_Recommendation/Sim_matrix_word2vec.csv

```

5. Download GoogleNews-vectors-negative300

6. start app.py


## FrontEnd Setup

1. Direct to NodeFrontend.

2. Install node modules

```sh
npm install
```

3. Start the client

```sh
npm start
```

## Image Server Setup

Download dataset and create an http server at port 5001.

http://images.cocodataset.org/zips/val2017.zip

```
python -m http.server 5001
```

## Screenshot

![](gif/gif.gif)
