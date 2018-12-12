# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 13:07
# @Author  : kean
# @Email   : 
# @File    : logger.py
# @Software: PyCharm


# Format	Description
# %(name)s	Name of the logger (logging channel).
# %(levelno)s	Numeric logging level for the message (DEBUG, INFO, WARNING, ERROR, CRITICAL).
# %(levelname)s	Text logging level for the message ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
# %(pathname)s	Full pathname of the source file where the logging call was issued (if available).
# %(filename)s	Filename portion of pathname.
# %(module)s	Module (name portion of filename).
# %(funcName)s	Name of function containing the logging call.
# %(lineno)d	Source line number where the logging call was issued (if available).
# %(created)f	Time when the LogRecord was created (as returned by time.time()).
# %(relativeCreated)d	Time in milliseconds when the LogRecord was created, relative to the time the logging module was loaded.
# %(asctime)s	Human-readable time when the LogRecord was created. By default this is of the form “2003-07-08 16:49:45,896” (the numbers after the comma are millisecond portion of the time).
# %(msecs)d	Millisecond portion of the time when the LogRecord was created.
# %(thread)d	Thread ID (if available).
# %(threadName)s	Thread name (if available).
# %(process)d	Process ID (if available).
# %(message)s	The logged message, computed as msg % args.
import os
from io import StringIO as StringBuffer

log_capture_string = StringBuffer()
# log_capture_string.encoding = 'cp1251'

proj_dir = os.path.dirname(__file__)
_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(levelname)s %(asctime)s %(process)d %(message)s'
        },
        'detail': {
            'format': '%(levelname)s %(asctime)s %(process)d ' + ' %(module)s.%(funcName)s line:%(lineno)d  %(message)s',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            # 'formatter': 'simple'
            'formatter': 'detail',
            'stream': log_capture_string,
        },
        'console1': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            # 'formatter': 'simple'
            'formatter': 'detail',
            # 'stream': log_capture_string,
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detail',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            # 'maxBytes': 1024,
            # 'backupCount': 3,
            'when': 'midnight',
            'interval': 1,
            'filename': os.path.join(proj_dir, 'log/info.log')
        },
        'err_file': {
            'level': 'ERROR',
            'formatter': 'detail',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'when': 'midnight',
            'interval': 1,
            'filename': os.path.join(proj_dir, 'log/error.log')
        },
        'perf': {
            'level': 'INFO',
            'formatter': 'simple',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'when': 'midnight',
            'interval': 1,
            'filename': os.path.join(proj_dir, 'log/perf.log')
        },
        'track': {
            'level': 'INFO',
            'formatter': 'simple',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'when': 'midnight',
            'interval': 1,
            'filename': os.path.join(proj_dir, 'log/track.log')
        },

    },
    'loggers': {
        'default': {
            'level': 'DEBUG',
            'handlers': ['console', 'console1', 'file', 'err_file', 'perf', 'track']
        },
        'console': {
            'handlers': ['file', 'err_file'],
            'level': 'DEBUG'
        },
        'perf': {
            'handlers': ['perf'],
            'level': 'DEBUG',
            'propagate': False
        },
        'track': {
            'handlers': ['track'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# import logging
# from logging.handlers import TimedRotatingFileHandler
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# # create a file handler
# # handler = logging.FileHandler('log/question.log')
# handler = TimedRotatingFileHandler(skeleton_path + '/API/log/rose', when = "D",interval=1, backupCount=0)
# # 按每天来记录
# # handler.suffix = "%Y-%m-%d-%H%M%S.log"
# handler.suffix = "%Y-%m-%d.log"
# handler.setLevel(logging.DEBUG)
# # create a logging format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s ===> %(message)s')
# handler.setFormatter(formatter)
# # add the handlers to the logger
# logger.addHandler(handler)


import logging
import logging.config

# logging.config.dictConfig(config)
logging.config.dictConfig(_LOGGING)
logger = logging.getLogger('default')
# logger = logging.getLogger(__name__)
#
# from update_neo4j_data.mail import SendMail
# class MyLogger(SendMail):
#     def debug(self,e,add_info = True):
#         if add_info:
#             self._message_add(e)
#         logger.debug(e)
#     def info(self,e,add_info = True):
#         if add_info:
#             self._message_add(e)
#         logger.info(e)
#     def error(self, e, add_info=True):
#         if add_info:
#             self._message_add(e)
#         logger.error(e)
#     def _message_init(self):
#         self.message = ""

if __name__ == "__main__":
    logger.debug("======= 测试=========")
    logger.info("======= 测试=========")
    logger.error("======= 测试=========")

    log_contents = log_capture_string.getvalue()
    log_capture_string.close()
    print(log_contents.lower())
