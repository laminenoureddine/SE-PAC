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
#$DOCKERCOMMAND build -t madlab:5000/inc_pc_tr_ts .
$DOCKERCOMMAND build --no-cache=true -t inc_pc_tr_ts .
#$DOCKERCOMMAND build -t inc_pc_tr_ts .

$DOCKERCOMMAND save inc_pc_tr_ts > inc_pc_tr_ts.tar
#scp inc_pc_tr_ts.tar lct-n-2:~
rsync -a --progress inc_pc_tr_ts.tar lct-n-2:~

#sudo docker push madlab:5000/inc_pc_tr_ts:latest
