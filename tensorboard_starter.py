import time
import os
import sys
import subprocess
import argparse

def checkNotFinish(popen_list):
    for eachpopen in popen_list:
        if eachpopen.poll() == None:
            return True
    return False

parser = argparse.ArgumentParser(description='RL')

parser.add_argument('--port', type=int, default=6006,
                    help='port to run the server on (default: 8097)')

parser.add_argument('--env_name', type=str, default="HalfCheetah-v2")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

base_command = "tensorboard --logdir="

for d in os.listdir("./log/{}/{}".format(args.env_name, args.seed)):
    print(d)
    base_command+="{0}-{1}-{2}:./log/{0}/{1}/{2},".format(args.env_name, args.seed, d)

base_command = base_command[:-1]
base_command = base_command+" --port {}".format(args.port)

p = subprocess.Popen( base_command, shell=True)


while True :
    if p.poll() == None:
        break

