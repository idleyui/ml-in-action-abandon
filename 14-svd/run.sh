#!/usr/bin/env bash
scp matrix_recommend.py colfax:/home/u22520/fromwin/14-svd/matrix_recommend.py
ssh colfax 'cd /home/u22520/fromwin/14-svd; qsub run.sh'
