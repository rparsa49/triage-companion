# How to run the triage companion

1. Navigate to your project folder

Clone this git repository and open a terminal (macOS/Linux) or Command Prompt/PowerShell (Windows) and change to your project directory:
```
cd path/to/cloned/repo
```
2. Create the virtual environment

Use the built-in venv module. 

macOS / Linux
```
python3 -m venv venv
```

Windows
```
py -m venv venv
```

This creates a folder named venv (you can choose a different name) containing the isolated Python environment.

3. Activate the virtual environment

Activating ensures that any python or pip commands refer to this environment. 
Python Packaging

macOS / Linux (bash/zsh)
```
source venv/bin/activate
```

Windows (Command Prompt)
```
venv\Scripts\activate.bat
```

Windows (PowerShell)
```
.\venv\Scripts\Activate.ps1
```

When activated, your prompt will typically include (venv) as a prefix, indicating you are inside the environment.

4. Install dependencies from requirements.txt

Assuming you’ve created requirements.txt with all needed packages:
```
pip install -r requirements.txt
```

This installs all the exact versions of libraries your project needs.

5. Run the application

Then you can execute your Flask app (or whatever entry script you have) while the virtual environment is active:
```
python app.py

```

You will see something along the lines of:
```
Starting Triage Companion Backend on [http://127.0.0.1:5000](http://127.0.0.1:5000)
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
```

Click the link that is provided to you in this. Do *not* open the index.html from your file explorer because changes won't be visible!