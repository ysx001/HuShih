from re import compile as _Re

_unicode_chr_splitter = _Re( '(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)' ).split

def split_unicode_chrs(text):
  return [ chr for chr in _unicode_chr_splitter( text ) if chr ]