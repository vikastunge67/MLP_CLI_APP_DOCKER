Containerized Machine Learning for Churn Prediction: From CLI to Flask API
                                                                             - Vikas(2022OCTVUGP0031)
1. Project Overview
This project involves building and managing a Python CLI application using a Multi-Layer Perceptron (MLP) to predict customer churn. The project incorporates machine learning, Dockerization, and Git for version control.

2. Project Objectives
•	Develop a machine learning model (MLP) for classification tasks.
•	Implement a CLI application in Python for evaluating predictive models.
•	Containerize the application using Docker for easy deployment.
•	Apply Git version control for collaborative development.
•	Optionally create a Flask application for API-based predictions.

3. Project Requirements
3.1 CLI Application Setup
•	Developed a Python CLI application to handle the following: 
o	Accept Input: Upload a CSV file containing features for backorder prediction.
o	Train MLP Models: Train SLP, SLMP, and MLMP classifiers on sample datasets.
o	Evaluate Data: Predict whether an item in the uploaded CSV will be backordered.
o	Output Results: Save predictions to a CSV file and display summary statistics.
3.2 Machine Learning Model
•	Model Structure: Implemented MLP models with configurable input, hidden, and output layers.
•	Training & Evaluation: Included early stopping and performance evaluation using accuracy, precision, recall, and F1-score metrics.
•	Hyperparameter Tuning: Provided functionality for tuning model parameters.
3.3 Docker Implementation
•	Created a Dockerfile to containerize the CLI application.
•	Used docker-compose.yaml to manage the Docker service.
•	Mounted volumes for seamless input/output file handling between the host and container.
3.4 Git Versioning System
•	Initialized a GitHub repository for project tracking.
•	Followed detailed commit message conventions.
3.5 Documentation
•	Created a README.md to document the project setup and execution.
•	Documented the MLP model architecture, Docker commands, and Git workflow.
3.6 Model Evaluation
•	Evaluated model performance using metrics like accuracy, precision, recall, and F1-score.
•	Included cross-validation for robust evaluation.
3.7 Bonus Task: Flask Application
•	Created a Flask application to expose model predictions via a REST API.
•	Dockerized the Flask application and exposed port 5000 on localhost.

4. Project Files
File Name	Description
train.py	Training and evaluation of MLP models.
cli.py	CLI application for model operations.
app.py	Flask application for predictions.
requirements.txt	Python package dependencies.
Dockerfile	Docker container setup instructions.
docker-compose.yaml	Docker service configuration.

5. Docker Setup and Execution
5.1 Build and Run Docker Containers
docker-compose build
docker-compose up
5.2 Test the CLI Application
docker exec -it churn_prediction-cli-app-1 bash
python cli.py
5.3 Test the Flask API
Endpoint: POST http://localhost:5000/predict
Example Command:
curl -X POST -F "file=@test_data.csv" http://localhost:5000/predict

6. Model Details
6.1 Model Architectures
•	SLP (Single Layer Perceptron)
•	SLMP (Single Layer Multi Perceptron)
•	MLMP (Multi-Layer Multi Perceptron)
6.2 Training Process
•	Optimizer: Adam
•	Loss Function: Binary Cross-Entropy
•	Early Stopping Mechanism

7. Results and Performance Evaluation
Metric	Value
Accuracy	0.91
Precision	0.89
Recall	0.87
F1-Score	0.88

8. Git Workflow
Branching Strategy:
•	main: Stable production-ready branch
•	development: Ongoing development features
Commit Message Format:
[feature/fix] Short description (Max 50 characters)
- Detailed description if required

9. Challenges and Solutions
Challenge	Solution
Torch installation issue	Added system dependencies in Dockerfile
Model convergence	Early stopping mechanism
Docker volume handling	Mounted data directory inside container
API response format issues	Ensured proper JSON response structure

10. Recommendations for Future Enhancements
•	Include comprehensive hyperparameter search.
•	Add logging for better debugging.
•	Extend the Flask API with more endpoints.
•	Integrate CI/CD pipelines.

11. Conclusion
This project successfully demonstrates the integration of machine learning, containerization, and version control. The CLI and Flask application provide a user-friendly interface for training and evaluating MLP models, offering a solid foundation for further improvements and real-world deployment.
