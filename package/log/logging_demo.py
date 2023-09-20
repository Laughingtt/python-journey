#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import datetime
import logging
import time
from logging.handlers import TimedRotatingFileHandler

class LogFilter:
    @staticmethod
    def info_filter(record):
        if record.levelname == 'INFO':
            return True
        return False

    @staticmethod
    def error_filter(record):
        if record.levelname == 'ERROR':
            return True
        return False


class TimeLoggerRolloverHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False,
                 atTime=None):
        super(TimeLoggerRolloverHandler, self).__init__(filename, when, interval, backupCount, encoding, delay, utc)

    def doRollover(self):
        """
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        log_type = 'info' if self.level == 20 else 'error'
        dfn = f"my.{datetime.datetime.now().strftime('%Y%m%d')}.{log_type}.log"
        self.baseFilename = dfn
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:
                    addend = -3600
                else:
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt


new_formatter = '[%(levelname)s]%(asctime)s:%(msecs)s.%(process)d,%(thread)d#>[%(funcName)s]:%(lineno)s  %(message)s'
fmt = logging.Formatter(new_formatter)
log_error_file = 'my.{}.error.log'.format(datetime.datetime.now().strftime('%Y%m%d'))
log_info_file = 'my.{}.info.log'.format(datetime.datetime.now().strftime('%Y%m%d'))

error_handler = TimeLoggerRolloverHandler(log_error_file, when='midnight')
error_handler.addFilter(LogFilter.error_filter)
error_handler.setFormatter(fmt)
error_handler.setLevel(logging.ERROR)

info_handel = TimeLoggerRolloverHandler(log_info_file, when='midnight')
info_handel.setFormatter(fmt)
info_handel.addFilter(LogFilter.info_filter)
info_handel.setLevel(logging.INFO)


log_info = logging.getLogger('info')
log_info.setLevel(logging.INFO)

log_info.addHandler(info_handel)
log_info.addHandler(error_handler)

if __name__ == '__main__':
    while 1:
        log_info.error('test error')
        time.sleep(1)
        log_info.info('test info')

