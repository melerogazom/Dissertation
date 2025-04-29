# A Comparative Study on the Use of Signature-Based and Handcrafted Feature Extraction Methods for Motion Tracking Exercises

## Description
This project is part of a dissertation focused on research, analysis, and implementation of the signature method and the handcrafted method for feature extraction purposes in human activity recognition (HAR). The repository includes Python scripts and documentation to support the research and experimentation process. The goal is to explore and compare the two feature extraction methods in four different upper body exercises: bicep curls, tricep pulldowns, front shoulder raises, and lateral shoulder raises.

## Folders

1. SignatureMethod: This folder contains two classifiers, one for binary class classification to detect exercise repetitions and another for multiclass classification to detect the repetitions and the exercises performed. It also contains the real_time_logger.py to input the movements manually to be later trained and tested and the real_time_system_performace.py, which displays an interface and detects the exercise currently being carried out by the motion detected, using a Witmotion device.
2. Comparison: This folder is for the handcrafted method. As above, it contains two classifiers, one for binary class classification to detect exercise repetitions and another for multiclass classification to detect the repetitions and the exercises performed. It also contains the co_real_time_logger.py to input the movements manually to be later trained and tested and the co_real_time_system_performce.py, which displays an interface and detects the exercise currently being carried out by the motion detected, using a Witmotion device.
3. Visualisation: This folder contains 5 line plots. These plots compare the F1 scores for different training data sizes, and the results are plotted using the F1 scores of the two feature extraction methods for each specific exercise.

## Requirements
To run this project, you need the following:

- **Python**: Version 3.x

## Contributing
Contributions to this project are welcome. If you have suggestions, bug fixes, or improvements, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or fix:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes and push the branch:
    ```bash
    git commit -m "Description of changes"
    git push origin feature-name
    ```
4. Open a pull request.

## Contact
For questions, feedback, or collaboration opportunities, please contact:

**Miguel Melero Gazo**  
Email: [mmg64@bath.ac.uk]  
[GitHub](https://github.com/melerogazom)