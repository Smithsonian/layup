from layup import routines


def test_hello_world() -> None:
    output = routines.hello_world()
    assert output == "Hello, World!"
