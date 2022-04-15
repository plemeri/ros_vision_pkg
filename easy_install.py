import os

user = os.popen('whoami').read()[:-1]
sudo_passwd = input('[sudo] password for {}:'.format(user))
distro = os.popen('rosversion -d').read()[:-1]
pwd = os.popen('pwd').read()[:-1]

os.system('echo {0} | sudo -S apt update'.format(sudo_passwd))
os.system('echo {0} | sudo -S apt install -y ros-{1}-vision-msgs ros-{1}-cv-bridge ros-{1}-pcl-conversions ros-{1}-pcl-ros ros-{1}-roslint ros-{1}-image-geometry ros-{1}-tf2-sensor-msgs ros-{1}-tf2-geometry-msgs'.format(sudo_passwd, distro))
os.system('echo {0} | sudo -S apt install -y libyaml-cpp-dev python3-pip python-pip'.format(sudo_passwd))
os.system('pip3 install --upgrade pip')
os.system('pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html')
os.system('pip3 install -r src/requirements.txt')
os.system('cd ros_vision_pkg')
os.system('catkin_make')
os.system('echo \"source {}\" >> ~/.bashrc'.format(os.path.join(pwd, 'devel', 'setup.bash')))