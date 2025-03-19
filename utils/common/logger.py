import logging
import colorlog

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()  # Log to console

        # Define color formatter
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s ==> %(message)s",
            log_colors={
                'DEBUG': 'bold_cyan',
                'INFO': 'bold_green',
                'WARNING': 'bold_yellow',
                'ERROR': 'bold_red',
                'CRITICAL': 'bold_red'
            }
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

LOGGER = get_logger()