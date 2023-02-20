## Code for computer vision

### Installing Dependencies run:
    pip install -r requirements.txt

if you're using windows Apriltag will not likely work, in that case try using
pupil_apriltag instead.

if any problem on Mac
try installing or re/installing xcode tools
### Tips
I recoommend adding `env` (virtual environment directory) under `IDP_L111/` so
you can run the files easily by doing:

    ./main.py
without having to activate the virtual envirionemtn everytime.

But you need to give the file execution privlage by doing:

    chmod +x main.py

## File layouts
    CV
    ├── archived_code
    |   └──   holds previously used codes
    |
    ├── calib_imgs
    |   └──   holds all theimages used for calibration etc
    |
    ├── calibration_data
    |   └──   holds all interation of calibration values
    |
    ├── moveToPoint_Offshoots
    |   └──   holds variations of moveToPoint files
    |
    ├── MQTT_test.py 
    |           - file for contolling the robot remotely
    |
    ├── README.md
    |           - this file
    |
    ├── calibration_clean.py
    |           - file for clibrating the camera
    |
    ├── moveToPoint_modular.py
    |           - the actual file used for navigating the robot
    |
    ├── points.txt
    |           - holds various points that is used for navigation etc..
    |
    └── requirements.txt
                - has all the depedencies



