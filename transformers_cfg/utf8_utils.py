from dataclasses import dataclass
from typing import Tuple

from dataclasses import dataclass


@dataclass
class PartialUTF8:
    """
    A data class representing the state of a partially decoded UTF-8 sequence.

    Attributes:
    - value (int): The current accumulated value of the partially decoded Unicode code point.
                   This attribute stores the bits that have been decoded so far. For a fully decoded
                   character or before any partial decoding has started, this would typically be `0`.

    - n_remain (int): The number of bytes remaining to complete the current UTF-8 encoded character.
                      A value of `-1` indicates that there is no ongoing partial decoding, i.e.,
                      either decoding has not started, or the last character was fully decoded.

    This class is used to handle situations where UTF-8 encoded data may end in the middle of a character
    sequence, allowing for the decoding process to be resumed when more data becomes available.
    """

    value: int = 0  # Default to 0, indicating no partial value accumulated
    n_remain: int = (
        -1
    )  # Default to -1, indicating no bytes are currently expected to complete the character

    def __hash__(self):
        return hash((self.value, self.n_remain))

    def __eq__(self, other):
        if not isinstance(other, PartialUTF8):
            return NotImplemented
        return self.value == other.value and self.n_remain == other.n_remain


from typing import List, Tuple
from functools import lru_cache


@lru_cache(maxsize=3000000)
def decode_utf8(
    src: bytes, partial_start: PartialUTF8
) -> Tuple[List[int], PartialUTF8]:
    # Lookup table for determining the total bytes based on the first byte's high 4 bits
    lookup = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4]
    pos = 0  # Position in the src bytes to start decoding from
    code_points = []  # List to store the decoded Unicode code points
    value = partial_start.value  # Start with any previously partial decoded value
    n_remain = partial_start.n_remain  # Number of bytes remaining from a partial decode

    # If there's a partial sequence left from last decode, try to continue decoding it
    while pos < len(src) and n_remain > 0:
        next_byte = src[pos]  # Get the next byte to process
        # Check if the continuation byte format is correct (`10xxxxxx`)
        if (next_byte >> 6) != 2:
            # If not, it's an invalid sequence. Abort and return a special error state.
            code_points = [0]
            return code_points, PartialUTF8(0, -1)

        # Accumulate the value by shifting left and adding the relevant 6 bits
        value = (value << 6) + (next_byte & 0x3F)
        pos += 1  # Move to the next byte
        n_remain -= 1  # Decrement the number of remaining bytes

    # If we've completed a partial sequence, add its value to the code points
    if partial_start.n_remain > 0 and n_remain == 0:
        code_points.append(value)

    # Process the rest of src as complete or new UTF-8 sequences
    while pos < len(src):
        first_byte = src[pos]  # Get the first byte of the next sequence
        highbits = first_byte >> 4  # Extract the high 4 bits for the lookup table
        n_remain = lookup[highbits] - 1  # Determine remaining bytes in this sequence

        # If lookup returns an invalid number, it's an invalid sequence. Abort.
        if n_remain < 0:
            # raise ValueError("Invalid UTF-8 sequence")
            code_points = [0]
            return code_points, PartialUTF8(0, -1)

        # Calculate the mask to isolate significant bits from the first byte
        mask = (1 << (7 - n_remain)) - 1
        value = first_byte & mask  # Apply the mask to get the initial value
        pos += 1  # Move to the next byte

        # Process the continuation bytes
        while pos < len(src) and n_remain > 0:
            next_byte = src[pos]
            # Shift the accumulated value and add the next 6 significant bits
            value = (value << 6) + (next_byte & 0x3F)
            pos += 1  # Move to the next byte
            n_remain -= 1  # Decrement the number of remaining bytes

        # If the sequence is complete, add its decoded value to the code points
        if n_remain == 0:
            code_points.append(value)

    # # Append a terminating value to indicate the end (following llama-cpp implementation)
    # code_points.append(0)
    # the following line is crucial for LRU cache to work, as it reset to the initial state
    if n_remain == 0:
        n_remain = -1
        value = 0

    # Return the decoded code points and the state of any partial decoding
    return code_points, PartialUTF8(value, n_remain)


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

    print("-" * 50)

    # Example usage:
    utf8_bytes = b"\xe2\x82\xacHello"  # UTF-8 encoded string (Euro symbol + "Hello")
    partial_start = PartialUTF8()  # Assuming start with no partial sequence
    code_points, partial_utf8 = decode_utf8(utf8_bytes, partial_start)

    print("Code Points:", code_points)
    print("Remaining UTF-8 State:", partial_utf8.value, partial_utf8.n_remain)
