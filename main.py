#!encoding=utf8
#created in 2018/6/9 cyh
import os
import pdb
import sys
import math
import time
import random
import model
import readdata
import yaml
import numpy as np
import tensorflow as tf
import argparse
tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":

  # Read config
  config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
  params = config["params"]
  data = config["data"]
  Dir = config["save"]

  # Read the inputted args
  parser = argparse.ArgumentParser()
  parser.add_argument("--num", "-n", type=int, default=0, help="The test number") 
  parser.add_argument("--mode", "-m", type=str, default="123", help="The running mode") 
  arg = parser.parse_args()
  mode = arg.mode
  testnum = arg.num

  for testnum in range(15):
    model_dir = os.path.join(Dir, f"{testnum}")
    # Model buildding
    name = tf.feature_column.numeric_column(key="name", shape=[1])
    left = tf.feature_column.numeric_column(key="left", shape=[36,60])
    right = tf.feature_column.numeric_column(key="right", shape=[36,60])
    head = tf.feature_column.numeric_column(key="head", shape=[6])
    label = tf.feature_column.numeric_column(key="label", shape=[6])
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True

    print("Bulid Model...")
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    start = time.clock()
    estimator = tf.estimator.Estimator(
      model_fn=model.ARModel,
      model_dir=model_dir,
      config=tf.estimator.RunConfig(session_config=tfconfig, save_checkpoints_steps=1000000, keep_checkpoint_max=None),
      params={
        'left': left,
        'right': right,
        'head': head,
        'name': name,
        'label': label
      }
    )

    end = time.clock()
    print("Build Model OK, used time %f s" % (end-start))
    if '1' in mode: 
      for i in range(40):
        sys.stdout.flush()
        
        start = end
        print("Train Model...")
        estimator.train(
          input_fn = lambda: readdata.ReadData(data["label"], data["root"], testnum)[0].repeat(params["test_steps"]).batch(params["batch_size"])
        )
        end = time.clock()
        print("Train Model OK, used time %f s" % (end-start))
        start = time.clock()

        print("Evaluate Model")
        result = estimator.evaluate(
          input_fn = lambda: readdata.ReadData(data["label"], data["root"], testnum)[1].batch(1)
        )
        end = time.clock()
        print(f"[{i}],Evaluate Model OK,used time {((end-start)/3600):.3f}h")
        print(f"[{i}],avg: {result['avg']}")
        print(f"[{i}],left:   {result['left']}")
        print(f"[{i}],right:,   {result['right']}")
        print(f"[{i}],choose:,  {result['choose']}")
        print(f"[{i}],accuracy:,  {result['accuracy']}") 


    if '2' in mode: 
      start = time.clock()
      print("Predict Model")
      results = estimator.predict(
        input_fn = lambda: readdata.ReadData(data["label"], data["root"], testnum)[1].batch(1)
      )
      for result in results:
        print(result["name"], result["left"], result["left_ac"]*180/math.pi, result["right_ac"]*180/math.pi, result["choose"])
      
      end = time.clock()
      print("Predict Model OK,used time %f s" % (end - start))
    
    if '3' in mode: 
      start = time.clock()
      print("Evaluate Model")
      result = estimator.evaluate(
        input_fn = lambda: readdata.ReadData(data["label"], data["root"], testnum)[1].batch(1)
      )
      end = time.clock()
      print("Evaluate Model OK,used time %f s" % (end - start))
      

      print("avg:",   result["avg"])
      print("left:",    result["left"])
      print("right:",   result["right"])
      print("choose:",  result["choose"])
      print("accuracy:",  result["accuracy"])
