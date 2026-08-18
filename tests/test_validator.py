from video_validator import validate_video


def test_missing_video():

    valid, message = validate_video(
        "file_that_does_not_exist.mp4"
    )

    assert valid is False
    assert "does not exist" in message