import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import logging
from my_log import get_default_logger


class Test(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.logger: logging.Logger = get_default_logger()
    
    def test_1(self):
        self.logger.debug('debug信息')
        self.logger.info('info,一般的信息输出')
        self.logger.warning('waring，用来用来打印警告信息')
        self.logger.error('error，一般用来打印一些错误信息')
        self.logger.critical('critical，用来打印一些致命的错误信息，等级最高')


if __name__ == "__main__":
    unittest.main()