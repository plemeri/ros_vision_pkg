import os

user = os.popen('whoami').read()[:-1]
sudo_passwd = input('[sudo] password for {}:'.format(user))
pwd = os.popen('pwd').read()[:-1]

os.system('echo {0} | sudo -S apt install docker.io'.format(sudo_passwd))
os.system('systemctl start docker'.format(sudo_passwd))
os.system('systemctl enable docker'.format(sudo_passwd))
os.system('echo {0} | sudo usermod -aG docker $USER'.format(sudo_passwd))
os.system('distribution=$(. /etc/os-release;echo $ID$VERSION_ID)')
os.system('curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -')
os.system('curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list')
os.system('echo {0} | sudo -S apt-get update && sudo apt-get install -y nvidia-docker2 nvidia-container-toolkit')
os.system('echo {0} | sudo -S systemctl restart docker')
os.system('export my_group=$(id -gn)')
os.system('newgrp docker')
os.system('newgrp $my_group')
os.system('docker pull ros:melodic')
os.system('docker run --gpus all --ipc=host --network host --name ros_vision_pkg --volume {0}:/projects -it ros:melodic'.format(pwd))