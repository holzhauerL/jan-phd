import os                       # file operations
import sys                      # standard output
from datetime import datetime   # date and time operations

class Logger():
    """
    Class to provide the functionality to log all printed outputs into a log file.
    """
    def __init__(self, log=True):
        """
        Initializes the loggable class.
        :param log: Whether to create a log file. Defaults to True.
        """
        self.DELIMITER = "-" * 80
        self.log = log
        self.log_path = None

        if self.log:
            self._log_create()
    
    def _log_create(self):
        """
        Creates the log file with the current date and time as the name.
        """
        # Create the name of the log file with the current date and time, 'log' as a prefix and '.txt' as a suffix
        log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        # Create the folder 'log' if it does not exist
        if not os.path.exists(os.path.join(os.getcwd(), 'log')):
            os.makedirs(os.path.join(os.getcwd(), 'log'))
        # Create the full path of the log file
        self.log_path = os.path.join(os.getcwd(), 'log', log_filename)
        # Create the log file
        with open(self.log_path, 'w') as log_file:
            output = f"Log file created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
            print(output)
            log_file.write(output)

    def log_print(self, func):
        """
        Decorator to save all printed outputs into the created log. Only works if the log attribute is set to True.
        """
        def wrapper(*args, **kwargs):
            if not self.log:
                return func(*args, **kwargs)

            original_stdout = sys.stdout
            # Check if the log file exists
            if not os.path.exists(self.log_path):
                self._log_create()
            with open(self.log_path, 'a') as log_file:
                log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Function: {func.__name__}\n")
                log_file.write(self.DELIMITER + "\n")
                sys.stdout = log_file
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    log_file.write(f"Error: {type(e)}\n")
                    log_file.write(f"Content: {e}\n")
                    raise e
                finally:
                    log_file.write(self.DELIMITER + "\n\n")
                    sys.stdout = original_stdout
            return result
        return wrapper