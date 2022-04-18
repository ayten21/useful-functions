import zipfile
def unzip_data(filename):
  
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()
    
    
  

import datetime
def create_tensorboard_callback(dir_name, model_name):
  """ Creates a Tensorboard Callback to store log files;
  Stores log files with the filepath:
  'dir_name/model_name/current_datetime/' """

  log_dir = dir_name + '/' + model_name + '/' + datetime.datetime.now().strftime(' %Y %m %d - %H %M %S')
  tensorboard_callback = tf.keras.callbacks.TensorBoard( log_dir = log_dir )
  print(f'Saving Tensorboard log files to : {log_dir}')
  return tensorboard_callback
