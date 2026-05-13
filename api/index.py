import os
import sys
import importlib.util

# Locate the project's Flask app file
repo_root = os.getcwd()
app_path = os.path.join(repo_root, "project", "Random_forest", "app.py")

spec = importlib.util.spec_from_file_location("rf_app", app_path)
rf_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf_app)

# Expose Flask app object for the serverless runtime
app = rf_app.app
