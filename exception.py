class ValueEmptyException(Exception):
    """
    空值异常
    """

    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self._msg


class ValueValidException(Exception):
    """
    值验证
    """

    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self._msg


class ResourceException(Exception):
    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self._msg
