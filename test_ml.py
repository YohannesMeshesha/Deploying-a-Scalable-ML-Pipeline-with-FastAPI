def test_one():
    """
    Test that True is True.
    """
    assert True


def test_two():
    """
    Test that 1 equals 1.
    """
    assert 1 == 1


def test_three():
    """
    Test that a non-empty list has a length greater than 0.
    """
    sample_list = [1, 2, 3]
    assert len(sample_list) > 0
