import subprocess
import sys
import os

def activate_environment(activate_script):
    """
    Activate a Python virtual environment.

    :param activate_script: Path to the activation script of the virtual environment.
    """
    if sys.platform.startswith('win'):
        # For Windows
        subprocess.call([activate_script], shell=True)
    else:
        # For Unix-like systems
        activate_cmd = "source " + activate_script
        os.system(activate_cmd)

def deactivate_environment():
    """
    Deactivate the currently active Python virtual environment.
    """
    if sys.platform.startswith('win'):
        # For Windows
        subprocess.call(["deactivate"], shell=True)
    else:
        # For Unix-like systems
        os.system("deactivate")