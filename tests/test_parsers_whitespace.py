"""Whitespace tolerance without weakening malformed-record validation."""
from hloc.utils import parsers as target
import pytest

@pytest.mark.parametrize('gap', ['   \n', '\t\n', ' \t \r\n'])
def test_image_list_ignores_whitespace_lines(tmp_path, gap):
    path = tmp_path / 'images.txt'
    path.write_text('a/b.jpg\n' + gap + 'c.jpg\n')
    assert target.parse_image_list(path) == ['a/b.jpg', 'c.jpg']

def test_image_list_ignores_indented_comments(tmp_path):
    path = tmp_path / 'images.txt'
    path.write_text('  # comment\na.jpg\n\t# another\nb.jpg\n')
    assert target.parse_image_list(path) == ['a.jpg', 'b.jpg']

@pytest.mark.parametrize('gap', ['   \n', '\t\n', ' \t \r\n'])
def test_retrieval_ignores_whitespace_lines(tmp_path, gap):
    path = tmp_path / 'pairs.txt'
    path.write_text('q/a.jpg r/b.jpg\n' + gap + 'q/a.jpg r/c.jpg\n')
    assert target.parse_retrieval(path) == {'q/a.jpg': ['r/b.jpg', 'r/c.jpg']}

def test_malformed_retrieval_is_still_rejected(tmp_path):
    path = tmp_path / 'pairs.txt'
    path.write_text('only-one-name\n')
    with pytest.raises(ValueError):
        target.parse_retrieval(path)

def test_normal_image_and_retrieval_inputs(tmp_path):
    module = target
    path = tmp_path / 'images.txt'
    path.write_text('# comment\n\na.jpg\nb.jpg\n')
    assert module.parse_image_list(path) == ['a.jpg', 'b.jpg']
    path.write_text('a.jpg b.jpg\na.jpg c.jpg\n')
    assert module.parse_retrieval(path) == {'a.jpg': ['b.jpg', 'c.jpg']}
