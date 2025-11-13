from pathlib import Path
from flask import Flask, render_template, request, jsonify
import requests
from urllib.parse import urljoin

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))  # หรือ "templates" ตามโครงสร้างจริง


# API configuration
API_BASE_URL = 'https://api-face-recognition-chi.vercel.app'

def get_employees():
    """Fetch employees from the API"""
    try:
        response = requests.get(urljoin(API_BASE_URL, '/employees'))
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            print(f"API request failed with status {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching employees: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/employee')
def employee():
    employees = get_employees()  # Get from your API
    return render_template('employee.html', employees=employees)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)