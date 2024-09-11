# Swarm


## Setup

- `git clone git@github.com:camoverride/swarm.git`
- `cd swarm`

 Create a virtual environment with `--system-site-packages` so we get the `picamera` package:

- `python -m venv --system-site-packages .venv`
- `source .venv/bin/activate`

Install this package for installing `dlib`:

- `pip install setuptools`

Install `cmake` which is requied by `dlib` which is in turn required by `face_recognition`:

- `sudo apt update`
- `sudo apt install cmake`
- `sudo apt install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev`
- `pip install dlib -vvv`

Install remaining requirements:

- `pip install -r requirements.txt`

Hide the cursor:

- `sudo apt-get install unclutter`


## Test

- `python display.py`


## Run in Production

Create a service with *systemd*:

- `mkdir -p ~/.config/systemd/user`
- `cat display.service > ~/.config/systemd/user/display.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`

Start it on boot:

- `sudo loginctl enable-linger pi`

Get the status:

- `systemctl --user start display.service`

Get the logs:

- `journalctl --user -u display.service`


## TODO

- [X] integrate picam
- [ ] implement dynamic alpha (should be high, like 0.8, then drop quickly to 0.1 over ~10 iterations)
- [ ] play with `face_memory` parameter
