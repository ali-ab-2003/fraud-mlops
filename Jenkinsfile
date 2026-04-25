pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Lint') {
            steps {
                sh 'flake8 .'
            }
        }

        stage('Unit Tests') {
            steps {
                sh 'pytest tests/'
            }
        }

        stage('Data Validation') {
            steps {
                sh 'python ci_data_validation.py'
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -f Dockerfile.training -t fraud-training .'
                sh 'docker build -f Dockerfile.inference -t fraud-inference .'
            }
        }

        stage('Kubeflow Trigger') {
            steps {
                sh 'python fraud_pipeline.py'
            }
        }

        stage('Monitoring') {
            steps {
                sh 'python monitoring/monitor.py'
            }
        }
    }
}
