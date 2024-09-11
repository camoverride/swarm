# Swarm


## Setup

- `git clone git@github.com:camoverride/swarm.git`
- `cd swarm`
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install setuptools` (for `face_recognition` models)
- `pip install -r requirements.txt`

Hide the cursor:

- `sudo apt-get install unclutter`


## Run in Production

Create a service with *systemd*:

- `mkdir -p ~/.config/systemd/user`
- `cat display.service ~/.config/systemd/user/display.service`

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

- [ ] integrate picam
- [ ] implement dynamic alpha (should be high, like 0.8, then drop quickly to 0.1 over ~10 iterations)
- [ ] play with `face_memory` parameter
