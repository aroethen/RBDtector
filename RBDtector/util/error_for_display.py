
class ErrorForDisplay(Exception):
    """
    Custom error class for all errors that should display an error message to the user.
    Best usage: First log the original error, then raise ErrorForDisplay(custom_error_message) from OriginalError
    """
    def __init__(self, display_message: str):
        """
        Constructor of ErrorForDisplay.
        :param display_message: Message to be displayed to user
        """
        self._display_message = display_message

    def __str__(self):
        return self._display_message
