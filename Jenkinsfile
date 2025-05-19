pipeline {
  agent any

  stages {
    stage('Checkout') {
      steps {
        git url: 'https://github.com/Ausland3r/CapaBara2.0.git', branch: 'main'
      }
    }

    stage('Inject .env') {
      steps {
        withCredentials([
          string(credentialsId: 'GITHUB_TOKEN', variable: 'GH_TOKEN'),
          string(credentialsId: 'GITHUB_REPOS', variable: 'GH_REPOS')
        ]) {
          writeFile file: '.env', text: """
GITHUB_TOKEN=${GH_TOKEN}
GITHUB_REPOS=${GH_REPOS}
"""
        }
      }
    }

    stage('Setup Python & Run') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt
          nohup python app.py > dash.log 2>&1 &
        '''
      }
    }
  }

  post {
    success { echo "Dash поднят, см. dash.log" }
    failure { echo "Ошибка" }
  }
}
