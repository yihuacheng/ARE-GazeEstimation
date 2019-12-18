#!/bin/bash
set -e
curpath=`pwd`
eval `cat config.ini`
export CUDA_VISIBLE_DEVICES=1
modelpath=${curpath}'/'${MODELPATH}
logpath=${curpath}'/'${LOGPATH}/$1

echo "Create the log file"
if [ ! -d $logpath ];then
	mkdir $logpath
else
	echo ${logpath}"is exist"
#	exit -1
fi

for arg in ${ARGS[@]}
do
	echo "Create the model folder"${arg}

	if [ ! -d ${modelpath}'/'${arg} ];then
		mkdir ${modelpath}'/'${arg}
	else
		echo ${modelpath}'/'${arg}" is exist"
#		exit -1
	fi

	echo "Train the model"
	
	/usr/bin/python3 main.py ${DATASET} ${arg} ${MODE} ${modelpath}'/'${arg} >> ${logpath}'/'${arg}  2>&1
done

mv  ${modelpath} ${logpath}"/"
