#! /usr/bin/python3

import os

tar_names = os.listdir('./bucket/data')
for file_name in tar_names:
	os.system('tar -xvf ./bucket/data/' + file_name)
