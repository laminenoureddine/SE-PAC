#!/bin/bash
DOCKERCOMMAND=""

if groups $USER  | grep &>/dev/null '\bdocker\b'
  then
    DOCKERCOMMAND="docker"
  else
    DOCKERCOMMAND="sudo docker"
fi

$DOCKERCOMMAND rm -f $($DOCKERCOMMAND ps -aq)
$DOCKERCOMMAND rmi -f $($DOCKERCOMMAND images -a)
$DOCKERCOMMAND volume prune
sudo rm -f /var/lib/docker/overlay2/*
sudo rm -f /var/lib/docker/tmp/docker*
#$DOCKERCOMMAND build -t madlab:5000/mds .
#$DOCKERCOMMAND build --no-cache=true -t mds .
$DOCKERCOMMAND build -t mds .

$DOCKERCOMMAND save mds > mds.tar
scp mds.tar lct-n-2:~
#sudo docker push madlab:5000/mds:latest