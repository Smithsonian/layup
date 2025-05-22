from layup import routines


def test_hello_world() -> None:
    output = routines.hello_world()
    assert output == "Hello, World!"


def test_autodiff_hello_world() -> None:
    output = routines.autodiff_hello_world()
    assert output == "u = 8.19315;du/dx = 5.25"
