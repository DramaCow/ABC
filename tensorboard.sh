addr=`ip addr | grep inet | grep -v inet6 | awk '{print $2}' | sed 's/...$//' | sed -n 2p`
printf "Tensorboard running on address: %s\n" $addr
tensorboard --host $addr --logdir=_logs --port 6006