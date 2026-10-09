class VideoError(Exception):
    """Base class for video processing errors."""


class VideoOpenError(VideoError):
    """The video file could not be opened or decoded."""


class EmptyVideoError(VideoError):
    """The video contains no readable frames."""


class UnsupportedFormatError(VideoError):
    """File extension is not in the allowed list."""


class FileTooLargeError(VideoError):
    """Uploaded file exceeds the size limit."""
