import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hloc.utils.read_write_model import Image, read_images_binary, write_images_binary


class BinaryImageNameTest(unittest.TestCase):
    def test_decode_independently_encoded_utf8_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.bin"
            name = "场景/été_🚀.jpg"
            header = struct.pack("<Qidddddddi", 1, 7, 1, 0, 0, 0, 2, 3, 4, 9)
            observations = struct.pack("<Qddq", 1, 1.25, 2.5, -1)
            path.write_bytes(header + name.encode("utf-8") + b"\x00" + observations)
            image = read_images_binary(path)[7]
            self.assertEqual(image.name, name)
            np.testing.assert_array_equal(image.xys, [[1.25, 2.5]])
            np.testing.assert_array_equal(image.point3D_ids, [-1])

    def test_writer_matches_binary_format_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.bin"
            name = "模型/naïve.png"
            image = Image(7, np.array([1., 0, 0, 0]), np.array([2., 3, 4]), 9,
                          name, np.empty((0, 2)), np.empty(0, dtype=np.int64))
            write_images_binary({7: image}, path)
            expected = struct.pack("<Qidddddddi", 1, 7, 1, 0, 0, 0, 2, 3, 4, 9)
            expected += name.encode("utf-8") + b"\x00" + struct.pack("<Q", 0)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(read_images_binary(path)[7].name, name)

    def test_ascii_names_and_multiple_records_remain_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.bin"
            images = {
                i: Image(i, np.array([1., 0, 0, 0]), np.zeros(3), 1, name,
                         np.array([[i, i + .5]]), np.array([-1], dtype=np.int64))
                for i, name in enumerate(["one.jpg", "two/sub.png"], 1)
            }
            write_images_binary(images, path)
            recovered = read_images_binary(path)
            self.assertEqual([x.name for x in recovered.values()], [x.name for x in images.values()])
            for i in images:
                np.testing.assert_array_equal(recovered[i].xys, images[i].xys)


if __name__ == "__main__":
    unittest.main()
