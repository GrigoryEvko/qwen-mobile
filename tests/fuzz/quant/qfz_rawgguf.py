"""A small independent walker of the GGUF byte layout, for the mutations of the reader fuzzers.

The walker does not use gguf-py, because gguf-py is the target. It gives
the position, the width and the role of each integer field of the header,
of the key-value section and of the tensor infos. A mutation then writes an
interesting value into a field with a role, for example an array count or a
tensor offset, which a random byte flip finds only by chance.

The walker stops at the first field that does not fit the bytes, thus it
also works on a file that a mutation already changed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

GGUF_MAGIC = 0x46554747
# The byte width of each scalar GGUF value type, by type number.
SCALAR_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9
INTERESTING_U64 = (0, 1, 2, 31, 32, 33, 255, 4096, 2**20, 2**24, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**40,
                   2**62, 2**63 - 1, 2**63, 2**64 - 64, 2**64 - 1)
INTERESTING_U32 = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 30, 31, 39, 40, 41, 255, 2**16, 2**31 - 1, 2**31,
                   2**32 - 1)


@dataclass(frozen=True)
class IntField:
    """One integer field of a GGUF file.

    Attributes:
        pos: The byte position
        width: 4 or 8
        role: The role, for example "n_tensors", "key_len", "array_count", "dim", "offset"
    """

    pos: int
    width: int
    role: str


def walk(data: bytes) -> list[IntField]:
    """Give the integer fields of a GGUF byte string, up to the first field that does not fit.

    Args:
        data: The bytes of a GGUF file, possibly malformed

    Returns:
        The fields in file order. Complexity is O(fields), with a cap of 200 000 array elements.
    """
    fields: list[IntField] = []
    n = len(data)

    def u32(pos: int) -> int | None:
        return struct.unpack_from("<I", data, pos)[0] if pos + 4 <= n else None

    def u64(pos: int) -> int | None:
        return struct.unpack_from("<Q", data, pos)[0] if pos + 8 <= n else None

    if u32(0) != GGUF_MAGIC or u32(4) is None:
        return fields
    fields.append(IntField(4, 4, "version"))
    n_tensors, n_kv = u64(8), u64(16)
    if n_tensors is None or n_kv is None:
        return fields
    fields.extend([IntField(8, 8, "n_tensors"), IntField(16, 8, "n_kv")])
    pos = 24
    budget = 200_000

    def string(pos: int, role: str) -> int | None:
        length = u64(pos)
        if length is None or pos + 8 + length > n:
            return None
        fields.append(IntField(pos, 8, role))
        return pos + 8 + length

    def value(pos: int, vtype: int, depth: int) -> int | None:
        nonlocal budget
        if vtype in SCALAR_WIDTH:
            return pos + SCALAR_WIDTH[vtype] if pos + SCALAR_WIDTH[vtype] <= n else None
        if vtype == STRING:
            return string(pos, "string_len")
        if vtype == ARRAY and depth < 2:
            sub, count = u32(pos), u64(pos + 4)
            if sub is None or count is None:
                return None
            fields.extend([IntField(pos, 4, "array_type"), IntField(pos + 4, 8, "array_count")])
            pos += 12
            for _ in range(count):
                budget -= 1
                if budget < 0:
                    return None
                nxt = value(pos, sub, depth + 1)
                if nxt is None:
                    return None
                pos = nxt
            return pos
        return None

    for _ in range(min(n_kv, 100_000)):
        nxt = string(pos, "key_len")
        if nxt is None:
            return fields
        vtype = u32(nxt)
        if vtype is None:
            return fields
        fields.append(IntField(nxt, 4, "value_type"))
        nxt = value(nxt + 4, vtype, 0)
        if nxt is None:
            return fields
        pos = nxt
    for _ in range(min(n_tensors, 100_000)):
        nxt = string(pos, "tensor_name_len")
        if nxt is None:
            return fields
        n_dims = u32(nxt)
        if n_dims is None or n_dims > 4:
            return fields
        fields.append(IntField(nxt, 4, "n_dims"))
        pos = nxt + 4
        for _ in range(n_dims):
            if u64(pos) is None:
                return fields
            fields.append(IntField(pos, 8, "dim"))
            pos += 8
        if u32(pos) is None or u64(pos + 4) is None:
            return fields
        fields.extend([IntField(pos, 4, "tensor_type"), IntField(pos + 4, 8, "offset")])
        pos += 12
    return fields


def put_int(data: bytearray, field: IntField, value: int) -> None:
    """Write ``value`` into a field, little endian, with the value cut to the width of the field.

    Args:
        data: The bytes, changed in place
        field: The field
        value: The value
    """
    fmt = "<I" if field.width == 4 else "<Q"
    struct.pack_into(fmt, data, field.pos, value & ((1 << (8 * field.width)) - 1))


def header(n_tensors: int, kv: list[tuple[str, bytes]]) -> bytes:
    """Give a GGUF v3 header with key-value pairs whose value bytes (type and payload) are given raw.

    Args:
        n_tensors: The tensor count to write
        kv: (key, raw bytes of the value type and the value)

    Returns:
        The bytes
    """
    out = struct.pack("<IIQQ", GGUF_MAGIC, 3, n_tensors, len(kv))
    for key, raw in kv:
        k = key.encode()
        out += struct.pack("<Q", len(k)) + k + raw
    return out


def string_value(text: str) -> bytes:
    """Give the raw bytes of a STRING value: the type, the length and the UTF-8 bytes."""
    b = text.encode()
    return struct.pack("<IQ", STRING, len(b)) + b


def array_value(sub_type: int, items: bytes, count: int) -> bytes:
    """Give the raw bytes of an ARRAY value: the type, the subtype, the count and the raw items."""
    return struct.pack("<IIQ", ARRAY, sub_type, count) + items


def build_file(kv: list[tuple[str, bytes]], tensors: list[tuple[str, object, int]], align: int = 32) -> bytes:
    """Give a complete GGUF file from raw values and numpy tensors, with the data aligned as gguf-py does.

    Args:
        kv: (key, raw bytes of the value type and the value)
        tensors: (name, numpy array in numpy order, ggml type number 0 for F32 or 1 for F16)
        align: The data alignment

    Returns:
        The bytes
    """
    infos, blobs, offset = b"", b"", 0
    for name, array, ggml_type in tensors:
        data = array.tobytes()
        infos += tensor_info(name, list(reversed(array.shape)), ggml_type, offset)
        pad = (-len(data)) % align
        blobs += data + b"\0" * pad
        offset += len(data) + pad
    head = header(len(tensors), kv) + infos
    return head + b"\0" * ((-len(head)) % align) + blobs


def tensor_info(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    """Give the raw bytes of one tensor info.

    Args:
        name: The tensor name
        dims: The dimensions (ne), at most 4
        ggml_type: The ggml type number
        offset: The data offset relative to the data section

    Returns:
        The bytes
    """
    b = name.encode()
    out = struct.pack("<Q", len(b)) + b + struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    return out + struct.pack("<IQ", ggml_type, offset & (2**64 - 1))
