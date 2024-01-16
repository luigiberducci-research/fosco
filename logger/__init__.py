import enum

from logger.logger import Logger


class LoggerType(enum.Enum):
    TEXT = enum.auto()
    AIM = enum.auto()


def make_logger(logger_type: LoggerType | str = None, **kwargs) -> Logger | None:
    if logger_type is None:
        logger_type = LoggerType.TEXT

    if isinstance(logger_type, str):
        logger_type = LoggerType[logger_type.upper()]

    if logger_type == LoggerType.AIM:
        from logger.logger_aim import AimLogger
        return AimLogger(**kwargs)
    elif logger_type == LoggerType.TEXT:
        from logger.logger_text import TextLogger
        return TextLogger(**kwargs)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
