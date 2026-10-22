# SINT
Simple AI architectures and algorithms testing for SMACv2

## How to install SMACv2 (Linux, tested on Arch)
In order to make `SMACv2` work you need to first install and set-up Blizzard's StarcraftII Headless engine:
```bash
cd ~
wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip
rm SC2.4.10.zip
```
The game is then installed in `~/StarCraftII/`.
You then need SMAC and SMACv2 maps (both are in [SMACv2 repo](https://github.com/oxwhirl/smacv2)):
```bash
mkdir -p ~/StarCraftII/Maps/SMAC_Maps
git clone --depth 1 https://github.com/oxwhirl/smacv2.git /tmp/smacv2
cp /tmp/smacv2/smacv2/env/starcraft2/maps/SMAC_Maps/*.SC2Map ~/StarCraftII/Maps/SMAC_Maps/
rm -rf /tmp/smacv2
```

## Set-up the python environment
You will need [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Mxncrou/SINT.git
cd SINT
uv sync
source .venv/bin/activate
```
Once this is done, please use the testing script:
```bash
python random_agents.py
```
Depending on your desktop environment you may face issues, as I am on Hyprland, I added these two lines for the render:
```py
import os
os.environ["SDL_VIDEODRIVER"] = "x11"
```
running this also works:
```bash
SDL_VIDEODRIVER=x11 python random_agents.py
```