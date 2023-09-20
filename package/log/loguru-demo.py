from loguru import logger


logger.add("data/log_{time}.log",
           rotation="1 days",
           encoding="utf-8",
           backtrace=True,
           diagnose=True,
           enqueue=True)

logger.debug("start logging ")