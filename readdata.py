import tensorflow as tf
from tensorflow import data
import numpy as np
import re
import sys
import glob
import os
import pdb

def OpenImg(filename, label):
  img_string = tf.read_file(filename["left"])
  img = tf.image.decode_image(img_string)
  left = tf.cast(tf.reshape(img, [36, 60]), tf.float64)
  
  img_string = tf.read_file(filename["right"])
  img = tf.image.decode_image(img_string)
  right = tf.cast(tf.reshape(img, [36, 60]), tf.float64)
  
  return {  "right": right/255, 
      "left":left/255,
      "label": tf.reshape(label,[6]),
      "name":filename["name"],
      "head":filename["head"]}, tf.reshape(label, [6])


def ReadData(path, root, number):
  print(f"[Read Data]: Test subject is {number}")
  trainlist = {}
  testlist = {}
  trainlabel = []
  testlabel = []

  persons = os.listdir(path)
  persons.sort()
  persons = [os.path.join(path, i) for i in persons]
  assert type(number) == int and number < 15, "Error in readdata.py "
 
  testset = persons.pop(number)
  trainsets = persons
  
  # Read files
  trainfiles = []
  for file_p in trainsets:
    lines = open(file_p).readlines()
    lines.pop(0)
    trainfiles.extend(lines)

  testfiles = open(testset).readlines()
  testfiles.pop(0)

  print(f"[Read Data]: Training image num is {len(trainfiles)}")
  print(f"[Read Data]: test image num is {len(testfiles)}")


  # Processing files to Dict
  traindicts = {}
  traindicts["right"] = [ os.path.join(root, line.strip().split()[0]) for line in trainfiles]
  traindicts["left"]  = [ os.path.join(root, line.strip().split()[1]) for line in trainfiles]
  traindicts["name"]  = [ count for count, line in enumerate(trainfiles)]
  traindicts["head"]  = [ list( map(eval, ",".join( line.strip().split()[5:7] ).split(",")) )
                             for line in trainfiles]
  
  trainlabel = [ list( map(eval, ",".join( line.strip().split()[3:5] ).split(",")) )
                             for line in trainfiles]

  testdicts = {}
  testdicts["right"] = [ os.path.join(root, line.strip().split()[0]) for line in testfiles]
  testdicts["left"]  = [ os.path.join(root, line.strip().split()[1]) for line in testfiles]
  testdicts["name"]  = [ count for count, line in enumerate(testfiles)]
  testdicts["head"]  = [ list(map(eval, ",".join(line.strip().split()[5:7]).split(",")))
                             for line in testfiles]

  testlabel = [ list(map(eval, ",".join(line.strip().split()[3:5]).split(",")))
                             for line in testfiles]

    
  trainset = data.Dataset.from_tensor_slices((traindicts, trainlabel))
  testset = data.Dataset.from_tensor_slices((testdicts, testlabel))

  trainset = trainset.shuffle(50000).map(OpenImg)
  testset = testset.map(OpenImg)
  return trainset,testset

if __name__ == "__main__":
  train,test = ReadData(sys.argv[1], sys.argv[2], int(sys.argv[3]))
  iterator = test.batch(1).make_one_shot_iterator()
  with tf.Session() as sess:
    for i in range(0,2):
      print(sess.run(iterator.get_next()))

