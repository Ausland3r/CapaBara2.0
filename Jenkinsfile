pipeline {
  agent any

  environment {
    GITHUB_TOKEN = credentials('github-token-id')
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Setup Python') {
      steps {
        bat 'python -m venv .venv'
        bat """
          call .venv\\Scripts\\activate
          pip install --upgrade pip
          pip install -r requirements.txt
        """
      }
    }

    stage('Run Dash App') {
      steps {
        bat 'start /B call .venv\\Scripts\\activate && python app.py'
      }
    }
  }

  post {
    success {
      echo "Dash-сервер запущен на порту 8050"
    }
    failure {
      echo "Что-то пошло не так"
    }
  }
}
