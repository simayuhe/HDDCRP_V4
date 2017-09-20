#!/bin/bash
#program:
#	this program help you to add file to git
#History:
#20170920
git add .
git commit -m "my test for sh"
git remote add origin git@github.com:simayuhe/HDDCRP_V4.git
git push -u origin master
exit 0
