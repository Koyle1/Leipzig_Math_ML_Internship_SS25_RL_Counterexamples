This repository contains the course work relating to the course "Mathemactis & Machine Leaning Internship" at Univeristy Leipzig. 
During the course the participants focussed on expanding prior research in the filed of reinforcement learning based counterexample search
conducted by Adam Zsolt Wagner in the paper "Constructions in combinatorics via neural networks".

How to run?
- Open Terminal
- change to the repository where these files are located (e.g. cd ~/path/to/MATH_ML)
- DO a)
    - conda env create -f conda_env.yml
    - conda activate math_ml
    - python train.py
- OR b)
    - change to the repository where these files are located (e.g ~/path/to/MATH_ML)
    - docker buildx build -t "math_ml" .
    - docker run "math_ml"

