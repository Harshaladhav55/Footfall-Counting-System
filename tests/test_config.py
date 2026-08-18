def calculate_crossing(previous_x, current_x, line_x):
    """
    Determine whether a person crossed the counting line.

    Returns:
        "IN"  -> crossed from left to right
        "OUT" -> crossed from right to left
        None  -> no crossing
    """

    if previous_x < line_x and current_x >= line_x:
        return "IN"

    if previous_x >= line_x and current_x < line_x:
        return "OUT"

    return None


def test_left_to_right_is_in():
    result = calculate_crossing(
        previous_x=400,
        current_x=600,
        line_x=500
    )

    assert result == "IN"


def test_right_to_left_is_out():
    result = calculate_crossing(
        previous_x=600,
        current_x=400,
        line_x=500
    )

    assert result == "OUT"


def test_no_crossing_left_side():
    result = calculate_crossing(
        previous_x=300,
        current_x=400,
        line_x=500
    )

    assert result is None


def test_no_crossing_right_side():
    result = calculate_crossing(
        previous_x=600,
        current_x=700,
        line_x=500
    )

    assert result is None 
