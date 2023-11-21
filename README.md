# Mining_Student_Behavior
This repo contains all the code used for the study, "Mining Student Behavior Patterns for Enhanced Performance Prediction in Introductory Programming: Keystroke Analysis and Ensemble Strategies"

## Overview

This repository contains code and data related to a study on programming keystroke data analysis. Programming keystroke data encapsulates intricate patterns that encode programmers' behaviors, offering insights for various applications such as grade prediction and understanding the traits of proficient and less proficient programmers.

## Files

- `compute_features.py`: Script for computing novel features by integrating factors like key presses, timestamps, source locations, and program words.
  
- `create_programs_from_keystrokes.py`: Script for creating programs from keystroke data.

- `data/`: Directory containing the programming keystroke dataset from the CS1 (CS 1400) course at a prominent U.S. university.

- `feature_selection.py`: Script implementing an ensemble-based feature selection algorithm to extract pertinent features.

- `__pycache__/`: Directory storing Python bytecode files.

- `train_final_model.py`: Script for training the final model after feature selection and hyperparameter optimization.

- `train_random_forest.py`: Script specifically for training the Random Forest classifier.

## Feature Engineering

Novel features are engineered based on prior research, intuition, and analysis of programming behaviors.

## Model Training

The study involves hyperparameter optimization and grade prediction using six classification and three regression algorithms. The algorithms include Random Forest, Decision Tree, Multilayered Perceptron, Random Sampling Consensus, and RUSBoost. Grades are categorized into three tiers: Low, Average, and High-grades.

## Results

Despite challenges such as class imbalance, plagiarism, limited per-assignment data, and the ceiling effect, the study achieves a commendable weighted F1 score of 73%. Additionally, an ensemble classification approach is proposed, combining Isolation Forest outlier detection with a trained and fine-tuned Random Forest classifier, resulting in an 80% accuracy on the test set.

## Contribution

This research contributes to the advancement of computer science education, particularly in enhancing the quality of education for undergraduates.

## Availability

The code and dataset used in this study will be made publicly available upon publication. We encourage collaboration and further exploration of the insights gained from this research.

---

**Note:** This README provides an overview of the project. For detailed information, refer to the documentation and code files in the repository.
