Here is a Python script that uses tkinter library for GUI. Pillow library is used for showing photos in the GUI.
The project demonstrates how OpenCV Python library's capabilites are used.

The script was developed and tested in Ubuntu 20.04 WSL (version 1) running on an Windows 10 21H2 machine

To run the script using a Python virtual environment please run the following commands inside the cloned repository:
    1. python3 -m virtualenv local
    2. source ./local/bin/activate
    3. pip3 install -r requirements.txt
    4. python3 image_process.py
    5. deactivate

Please note that tkinter library should be preinstalled in the Linux system. In case it is missing, install using command:
    sudo apt-get install python3-tk