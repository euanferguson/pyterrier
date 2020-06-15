import deprecation

from . import mavenresolver

stdout_ref = None
stderr_ref = None
TERRIER_PKG = "org.terrier"

@deprecation.deprecated(deprecated_in="0.1.3",
                        # remove_id="",
                        details="Use the logging(level) function instead")
def setup_logging(level):
    logging(level)

def logging(level):
    from jnius import autoclass
    autoclass("org.terrier.python.PTUtils").setLogLevel(level, None)
# make an alias
_logging = logging

def setup_jnius():
    ''' this methods adds more utility methods to Python version of Terrier classes '''
    from jnius import protocol_map  # , autoclass
    from . import check_version


    # IterablePosting

    def _iterableposting_next(self):
        ''' dunder method for iterating IterablePosting '''
        nextid = self.next()
        # 2147483647 is IP.EOL. fix this once static fields can be read from instances.
        if 2147483647 == nextid:
            raise StopIteration()
        return self

    protocol_map["org.terrier.structures.postings.IterablePosting"] = {
        '__iter__': lambda self: self,
        '__next__': lambda self: _iterableposting_next(self)
    }

    # Lexicon

    def _lexicon_getitem(self, term):
        ''' dunder method for accessing Lexicon '''
        rtr = self.getLexiconEntry(term)
        if rtr is None:
            raise KeyError()
        return rtr

    protocol_map["org.terrier.structures.Lexicon"] = {
        '__getitem__': _lexicon_getitem,
        '__contains__': lambda self, term: self.getLexiconEntry(term) is not None,
        '__len__': lambda self: self.numberOfEntries()
    }

    # Index

    def _has_positions(index):
        p = index.getInvertedIndex().getPostings(index.getLexicon().getLexiconEntry(0).getValue())
        return "getPositions" in dir(p)

    def _index_add(self, other):
        from . import autoclass
        fields_1 = self.getCollectionStatistics().getNumberOfFields()
        fields_2 = self.getCollectionStatistics().getNumberOfFields()
        if fields_1 != fields_2:
            raise ValueError("Cannot document-wise merge indices with different numbers of fields (%d vs %d)" % (fields_1, fields_2))
        blocks_1 = _has_positions(self)
        blocks_2 = _has_positions(other)
        if blocks_1 != blocks_2:
            raise ValueError("Cannot document-wise merge indices with and without positions (%r vs %r)" % (blocks_1, blocks_2))
        multiindex_cls = autoclass("org.terrier.realtime.multi.MultiIndex")
        return multiindex_cls([self, other], blocks_1, fields_1 > 0)

    protocol_map["org.terrier.structures.Index"] = {
        # this means that len(index) returns the number of documents in the index
        '__len__': lambda self: self.getCollectionStatistics().getNumberOfDocuments(),

        # document-wise composition of indices: adding more documents to an index, by merging two indices with 
        # different numbers of documents. This implemented by the overloading the `+` Python operator
        '__add__': _index_add
    }

    # MetaIndex
    if check_version(5.3):
        
        def _metaindex_getitem(self, item):
            import pandas as pd
            if isinstance(item, int):
                keys = self.getKeys()
                if item < 0 or item >= len(self):
                    raise IndexError("%d is out of range" % item)
                values = self.getAllItems(item)
                return pd.Series(values, index=keys, name=str(item))
            if isinstance(item, str):
                keys = self.getKeys()
                if not item in keys:
                    raise TypeError("%s is not a metaindex key (available: %s)" % (item, str(keys)))
                docids = list(range(0, len(self)))
                values = self.getItems(item, docids)
                return pd.Series(values, index=docids, name=item)

        def _metaindex_as_df(self):
            import pandas as pd
            rows=[]
            keys=self.getKeys()
            docids = list(range(0, len(self)))
            values=self.getItems(keys, docids)
            return pd.DataFrame(values, index=docids, columns=keys)

        protocol_map["org.terrier.structures.MetaIndex"] = {
            # this means that len(meta) returns the number of documents in the metaindex
            '__len__': lambda self: self.size(),

            # allows access to metaindex columns and rows as pandas Series
            '__getitem__' : _metaindex_getitem,

            # returns a copy of the meta index as a dataframe
            'as_df' : _metaindex_as_df
        }


def setup_terrier(file_path, terrier_version=None, helper_version=None, boot_packages=[]):
    """
    Download Terrier's jar file for the given version at the given file_path
    Called by pt.init()

    Args:
        file_path(str): Where to download
        terrier_version(str): Which version of Terrier - None is latest
        helper_version(str): Which version of the helper - None is latest
    """
    # If version is not specified, find newest and download it
    if terrier_version is None:
        terrier_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-assemblies")
    else:
        terrier_version = str(terrier_version) # just in case its a float
    # obtain the fat jar from Maven
    trJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-assemblies", terrier_version, file_path, "jar-with-dependencies")

    # now the helper classes
    if helper_version is None:
        helper_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-python-helper")
    else:
        helper_version = str(helper_version) # just in case its a float
    helperJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-python-helper", helper_version, file_path, "jar")

    classpath=[trJar, helperJar]
    for b in boot_packages:
        parts = b.split(":")
        if len(parts)  < 2 or len(parts) > 4:
            raise ValueError("Invalid format for package %s" % b)
        group = parts[0]
        pkg = parts[1]
        filetype = "jar"
        version = None
        if len(parts) > 2:
            version = parts[2]
            if len(parts) > 3:
                filetype = parts[3]
        #print((group, pkg, filetype, version))
        filename = mavenresolver.downloadfile(group, pkg, version, file_path, filetype)
        classpath.append(filename)

    return classpath

def is_binary(f):
    import io
    return isinstance(f, (io.RawIOBase, io.BufferedIOBase))

def redirect_stdouterr():
    from jnius import autoclass, PythonJavaClass, java_method

    # TODO: encodings may be a probem here
    class MyOut(PythonJavaClass):
        __javainterfaces__ = ['org.terrier.python.OutputStreamable']

        def __init__(self, pystream):
            super(MyOut, self).__init__()
            self.pystream = pystream
            self.binary = is_binary(pystream)

        @java_method('()V')
        def close(self):
            self.pystream.close()

        @java_method('()V')
        def flush(self):
            self.pystream.flush()

        @java_method('([B)V', name='write')
        def writeByteArray(self, byteArray):
            # TODO probably this could be faster.
            for c in byteArray:
                self.writeChar(c)

        @java_method('([BII)V', name='write')
        def writeByteArrayIntInt(self, byteArray, offset, length):
            # TODO probably this could be faster.
            for i in range(offset, offset + length):
                self.writeChar(byteArray[i])

        @java_method('(I)V', name='write')
        def writeChar(self, chara):
            if self.binary:
                return self.pystream.write(bytes([chara]))
            return self.pystream.write(chr(chara))

    # we need to hold lifetime references to stdout_ref/stderr_ref, to ensure
    # they arent GCd. This prevents a crash when Java callsback to  GCd py obj

    global stdout_ref
    global stderr_ref
    import sys
    stdout_ref = MyOut(sys.stdout)
    stderr_ref = MyOut(sys.stderr)
    jls = autoclass("java.lang.System")
    jls.setOut(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')(stdout_ref),
            signature="(Ljava/io/OutputStream;)V"))
    jls.setErr(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')(stderr_ref),
            signature="(Ljava/io/OutputStream;)V"))
