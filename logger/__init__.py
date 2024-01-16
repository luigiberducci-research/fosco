import enum

from logger.logger import Logger


class LoggerType(enum.Enum):
    AIM = enum.auto()

def make_logger(logger_type: LoggerType | str, **kwargs) -> Logger:
    if isinstance(logger_type, str):
        logger_type = LoggerType[logger_type.upper()]

    if logger_type == LoggerType.AIM:
        from logger.logger_aim import AimLogger
        return AimLogger(**kwargs)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")