# CAPA Generation Tool

This tool automatically generates warnings for changes made in pull requests.

## Project Context
Developed during the preparation of **Alexander E. Ilchuck's bachelor thesis** at:  
**SPbPU Institute of Computer Science and Cybersecurity (SPbPU ICSC)**

## Authors & Contributors
### Main Contributor
- **Alexander E. Ilchuck**  
  Student at SPbPU ICSC

### Advisor & Minor Contributor
- **Vladimir A. Parkhomenko**  
  Senior Lecturer at SPbPU ICSC

## Reference
*(to be updated after official publication)*  
When using this repository, please cite:  
[Современные технологии в разработке программного обеспечения](https://hsse.spbstu.ru/userfiles/files/1941_sovremennie_tehnologii_s_oblozhkoy.pdf), 2025

## Project Description
This bachelor's thesis project focuses on developing a system for automated analysis of commit history in GitHub repositories to identify potentially risky code changes and generate recommendations for Corrective and Preventive Actions (CAPA).

The system includes:

- Data collection and preprocessing from GitHub API and local repository clones
- Calculation of commit metrics including change volume, complexity, and static analysis
- Training a machine learning model for commit risk classification
- Generation of software development improvement recommendations
- Visualization of results and analytics through a Dash-based web dashboard

## Technologies
- Python 3.9+
- GitPython
- Requests
- scikit-learn, deep-forest
- Dash (Plotly)
- Static analysis: pylint, bandit, eslint, checkstyle

## Project Structure
- `repository_analysis.py` - module for repository data collection and analysis
- `ml_model.py` - implementation of commit risk classification model
- `recommendations.py` - CAPA recommendation generator
- `dashboard.py` - web interface for analytics visualization

## Setup and Run Instructions
1. Install dependencies from `requirements.txt`
2. Obtain GitHub token for API access
3. Run data collection and commit analysis script
4. Train classification model
5. Launch Dash web server to view results
