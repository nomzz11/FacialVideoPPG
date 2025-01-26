# Facial Video PPG - End of studies project

## Project description 

This project aims to create a PPG system allowing, using the camera of a mobile phone, to measure certain physiological constants such as the pulse.

## Run project
Install dependancies :
```sh
sudo apt update
sudo apt install python3.8
sudo apt install python3-pip
pip install -r requirements.txt 

sudo apt update
sudo apt upgrade
sudo apt install python3.10
sudo apt install python3.10-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
python3.10 -m pip install -r requirements.txt
```

Setup environment variables :
```sh
cp .env.example .env
```
## Update requirements
Make sure you have pipreqs installed
```sh
pip3 install pipreqs
```
To update requirements run the following :
```sh
pipreqs --force .
```