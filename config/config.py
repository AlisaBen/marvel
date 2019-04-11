import os
import configparser
import sys


def load_conf(conf_file):
    print(os.path.exists(conf_file))
    config = configparser.ConfigParser(allow_no_value=True,interpolation=configparser.ExtendedInterpolation())
    config.read(conf_file)
    # print(config)
    config.sections()  # 这个要写，不然parser会有问题
    # print(config.sections())
    return config['FM']


fm_conf = load_conf(sys.argv[1])
# TRAINING_PATH = fm_conf.get('TRAINING_PATH')
# TEST_PATH = fm_conf.get('TEST_PATH')
# SAVE_PATH = fm_conf.get('SAVE_PATH')
# GPU = fm_conf.getboolean('GPU')
# GPU_INDEX = fm_conf.getint('GPU_INDEX')
# VALIDATION_RATE = fm_conf.getfloat('VALIDATION_RATE')
# ONLINE_FLAG = fm_conf.getboolean('ONLINE_FLAG')
# OPTIMIZER = fm_conf.get('OPTIMIZER')
# LEARNING_RATE = fm_conf.getfloat('LEARNING_RATE')
# WEIGHT_DECAY = fm_conf.getfloat('WEIGHT_DECAY')
# NUM_EPOCHS = fm_conf.getint('NUM_EPOCHS')
# BATCH_SIZE = fm_conf.getint('BATCH_SIZE')
# FIELD_NUM = fm_conf.getint('FIELD_NUM')
# SPARSE_SPLIT = fm_conf.getint('SPARSE_SPLIT')
# FEATURE_SIZE = fm_conf.getint('FEATURE_SIZE')
# TRAINING = fm_conf.getboolean('TRAINING')


class Args:
    # fm_conf = load_conf(sys.argv[1])
    TRAINING_PATH = fm_conf.get('TRAINING_PATH')
    TEST_PATH = fm_conf.get('TEST_PATH')
    FINISH_SAVE_PATH = fm_conf.get('FINISH_SAVE_PATH')
    LIKE_SAVE_PATH = fm_conf.get('LIKE_SAVE_PATH')
    SUBMISSION_PATH = fm_conf.get('SUBMISSION_PATH')
    # GPU = fm_conf.getboolean('GPU')
    GPU_INDEX = fm_conf.getint('GPU_INDEX')
    LOSS = fm_conf.get('LOSS')
    VALIDATION_RATE = fm_conf.getfloat('VALIDATION_RATE')
    # ONLINE_FLAG = fm_conf.getboolean('ONLINE_FLAG')
    OPTIMIZER = fm_conf.get('OPTIMIZER')
    # LEARNING_RATE = fm_conf.getfloat('LEARNING_RATE')
    LIKE_LEARNING_RATE = fm_conf.getfloat('LIKE_LEARNING_RATE')
    FINISH_LEARNING_RATE = fm_conf.getfloat('FINISH_LEARNING_RATE')
    WEIGHT_DECAY = fm_conf.getfloat('WEIGHT_DECAY')
    FINISH_NUM_EPOCHS = fm_conf.getint('FINISH_NUM_EPOCHS')
    LIKE_NUM_EPOCHS = fm_conf.getint('LIKE_NUM_EPOCHS')
    # BATCH_SIZE = fm_conf.getint('BATCH_SIZE')
    LIKE_BATCH_SIZE = fm_conf.getint('LIKE_BATCH_SIZE')
    FINISH_BATCH_SIZE = fm_conf.getint('FINISH_BATCH_SIZE')
    FIELD_NUM = fm_conf.getint('FIELD_NUM')
    SPARSE_SPLIT = fm_conf.getint('SPARSE_SPLIT')
    # FEATURE_SIZE = fm_conf.getint('FEATURE_SIZE')
    # TRAINING = fm_conf.getboolean('TRAINING')
    # EMBEDDING_SIZE = fm_conf.getint('EMBEDDING_SIZE')
    SAVE_PARAMS_PATH_PREFIX = fm_conf.get('SAVE_PARAMS_PATH_PREFIX')
    TASK = fm_conf.get('TASK')
    CONFIG_NAME = fm_conf.get('CONFIG_NAME')
    # SAVE_MODEL_PATH = fm_conf.get('SAVE_MODEL_PATH')
    # DEEP_FM = fm_conf.getboolean('DEEP_FM')
    # MERGE_FILE_PATH = fm_conf.get('MERGE_FILE_PATH').split(',')
    # PROCESS_DATA_PATH = fm_conf.get('PROCESS_DATA_PATH').split(',')
    # PROCESS = fm_conf.get('PROCESS')
    LIKE_MODEL_PATH = fm_conf.get('LIKE_MODEL_PATH')
    FINISH_MODEL_PATH = fm_conf.get('FINISH_MODEL_PATH')

    FINISH_LAYER = fm_conf.get('FINISH_LAYER').split(',')
    LIKE_LAYER = fm_conf.get('LIKE_LAYER').split(',')
    DROPOUT_PROB = fm_conf.getfloat('DROPOUT_PROB')
    FINISH_DROPOUT_PROB = fm_conf.getfloat('FINISH_DROPOUT_PROB')
    LIKE_DROPOUT_PROB = fm_conf.getfloat('LIKE_DROPOUT_PROB')
    FINISH_EMBEDDING_SIZE = fm_conf.getint('FINISH_EMBEDDING_SIZE')
    LIKE_EMBEDDING_SIZE = fm_conf.getint('LIKE_EMBEDDING_SIZE')
    CONV1D_LAYER = fm_conf.get('CONV1D_LAYER').split(',')
    MODEL = fm_conf.get('MODEL')
    TEST = fm_conf.get('TEST')

for k, v in fm_conf.items():
    print('[%s = %s]' % (k, v))
    # if k == 'merge_file_path':
    #     ls = v.split(',')
    #     print(ls)
        # print(type(v))
        # print(v)


