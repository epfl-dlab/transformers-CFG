def decode_utf8_leading_char(src: bytes) -> tuple:
    first_byte = src[0]
    highbits = first_byte >> 4
    lookup = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4]
    char_len = lookup[highbits]

    # Extract the relevant bytes for the UTF-8 character
    utf8_char_bytes = src[:char_len]

    # Decode the character
    char = utf8_char_bytes.decode("utf-8")

    # Use ord() to convert the single character to its Unicode code point
    code_point = ord(char)

    # Remaining bytes
    remaining_bytes = src[char_len:]

    return code_point, remaining_bytes


def decode_utf8_string(utf8_bytes: bytes) -> list:
    code_points = []
    while utf8_bytes:
        code_point, utf8_bytes = decode_utf8_leading_char(utf8_bytes)
        code_points.append(code_point)
    return code_points


if __name__ == "__main__":
    # Given string
    my_string = "€Hello"  # The Euro symbol followed by "Hello"

    # Get UTF-8 encoded bytes
    utf8_bytes = my_string.encode("utf-8")

    assert utf8_bytes == b"\xe2\x82\xacHello"

    # Example usage with the Euro symbol followed by more characters
    code_point, remaining_bytes = decode_utf8_leading_char(utf8_bytes)

    print(f"Code Point: {code_point}")  # Expected Output: 8364 (Euro symbol)
    print(f"Remaining Bytes: {remaining_bytes}")  # Expected Output: b'Hello'

    # Example usage with the entire string
    code_points = decode_utf8_string(utf8_bytes)

    print(
        f"Code Points: {code_points}"
    )  # Expected Output: [8364, 72, 101, 108, 108, 111]
