'''Python Wrappers for Kaldi table IO (:kaldi:`io.html`)

In Kaldi the archive does not carry information about its contents and the user is required to 
use the proper Reader or Writer. This module follows this approach and provides wrappers for 
RandomAccess and Sequential readers,  and for the Writers. The classes are instantiated for 
each Kaldi type.

Internally, the wrappers define holders (:kaldi:`io.html#io_sec_holders`) for python types
and instantiates the regular Kaldi templates. In this way, the wrappers are 100% compatible with 
Kaldi and support using pipes and subprograms for inputs and outputs. 

The Python readers and writers implement the context api, and are fully usable with the Python 
`with` construct.

Examples:
    A matrix to text converter:
    
    .. code-block:: python
     
        with kaldi_io.SequentialBaseFloatMatrixReader('ark:mat.ark') as reader:
            for name,mat in reader:
                print name, mat

    A simple vector generator:
    
    .. code-block:: python
    
        with kaldi_io.Int32VectorWriter('ark:| gzip -c vec.ark.gz') as w:
            for len in xrange(10):
                vec = [len] * len
                w['vec_%d' %(len,)] = vec
    
Kaldi Reader classes
====================

Kaldi provides two types of reader: the Sequential reader which is akin to an iterator and the 
Random Access reader which is akin to a dict. Both work with piped data, thus the random access
readers may be required to read and store objects in memory until the proper one is found. More
information is in :kaldi:`io.html#io_sec_bloat`. 

Kaldi programs typically open one Sequential reader (e.g. for the features) and several RandomAccess 
readers. For each feature, the random access readers would be used to fetch auxiliary information, while 
ensuring that they pertain to the same utterance. This resemples a merge-sort merge phase and works well
if all the files are properly sorted. Citing :kaldi:`data_prep.html#data_prep_data_yourself`:

.. note::

    All of these files should be sorted. If they are not sorted, you will get errors when you run the scripts. In The Table concept we explain why this is needed. It has to do with the I/O framework; the ultimate reason for the sorting is to enable something equivalent to random-access lookup on a stream that doesn't support fseek(), such as a piped command. Many Kaldi programs are reading multiple pipes from other Kaldi commands, reading different types of object, and are doing something roughly comparable to merge-sort on the different inputs; merge-sort, of course, requires that the inputs be sorted. Be careful when you sort that you have the shell variable LC_ALL defined as "C", for example (in bash),

    export LC_ALL=C

    If you don't do this, the files will be sorted in an order that's different from how C++ sorts strings, and Kaldi will crash. You have been warned!

.. py:class:: DataTypeSequentialReader(rx_specifier)

    The SequentialReader mostly ressembles a Python iterator. Therefore it implements the 
    Iterator protocol:
    
    .. py:method:: __iter__()
        
        Returns self
        
    .. py:method:: next()
        
        :return: a tuple of:
            
            * key (string)
            * value (type is determined by the reader class)
    
    Moreover it provides a method to check whether the iterator is empty:
    
    .. py:method:: done()
    
        Returns `True` if the iterator is empty
    
    Kaldi uses a slightly different iteration protocol, which can be accessed using the functions:
    
    .. py:method:: _kaldi_next()
    
        Advance the iterator by one value
        
    .. py:method:: _kaldi_key()
        
        Returns the key of the cirrent value
        
    .. py:method:: _kaldi_value()
    
        Returns the current value (i.e. the value that will be returned on the next call 
        to :func:`next`)
    
    For resource management the classes implement:
    
    .. py:method:: close()
        
        Closes the reader.
    
    .. py:method:: is_open() 
    
        Returns `True` is the reader is opened and can be read from
        
    .. py:method:: __enter__()
    .. py:method:: __exit__()
        
        Implement the `with` context protocol 


.. py:class:: DataTypeRandomAccessReader(rx_specifier)
    
    The random access ressembles a Python dict - values are retrieved for a given key value.
    Therefore the rader acts in a dict-like manner:
    
    .. py:method:: __contains__(key)
    .. py:method:: has_key(key)
    
        Returns `True` if key is present in reader. Enabvles the use of the `in` operator.
        
    .. py:method:: __getitem__(key)
    .. py:method:: value(key)
    
        Returns the value associeted with key
        
    For resource management the classes implement:
    
    .. py:method:: close()
        
        Closes the reader.
    
    .. py:method:: is_open() 
    
        Returns `True` is the reader is opened and can be read from
        
    .. py:method:: __enter__()
    .. py:method:: __exit__()
        
        Implement the `with` context protocol 

.. py:class: DataTypeRandomAccessReaderMapped(data_rx_specifier, maping_rx_specifier)
    This class implement a random access reader whose keys have been mapped using the mapper.
    See :kaldi:`io.html#io_sec_mapped` for more explanation

Kaldi Writer class
==================

Th writer stores key-value pairs and thus ressembles a dict. However, unlike a dict 
no checks for key duplication are made. The writer will happily store all values using 
the same key, which may render them unusable. For best cooperation with KAldi, the keys 
should be written sorted in the `C order`.  


.. py:class:: DataTypeWriter(wx_specifier)
    .. py:method:: write(key, value)
    .. py:method:: __setitem__(key,value)
    
        Append to the file the value under key
    
    .. py:method:: flush()
    
        Flush the output stream.
        
    For resource management the classes implement:
    
    .. py:method:: close()
        
        Closes the writer.
    
    .. py:method:: is_open() 
    
        Returns `True` is the writer is opened and can be written to
        
    .. py:method:: __enter__()
    .. py:method:: __exit__()
        
        Implement the `with` context protocol   

Transformed Readers
===================

Very often the value read into Python would need to be further converted. The classes
`TransRA` and `TransSeq` take an appropriate reader and a function that will be used to 
transform all objects returned

    
Mapping between Kaldi and Python Objects
========================================

The readers and writers are named after the Kaldi type they access.

+--------------------+---------------------+-----------------------+-----------------------+
|     Kaldi Type     |  Read Python Type   | Writable Python Types |         Notes         |
|                    |                     |                       |                       |
+====================+=====================+=======================+=======================+
|Matrix              |NDArray of           |Any Python object      |BaseFloat is mapped to |
|                    |appropriate          |convertible to an      |either float32 (c's    |
|                    |DTYPE. Float32 and   |NDarray                |float) or float64 (c's |
|                    |Float64 are used for |                       |double) based on Kaldi |
|                    |float and double,    |                       |compile options        |
|                    |respectively.        |                       |                       |
+--------------------+---------------------+-----------------------+-----------------------+
|Vector              |1-dimensional NDarray|Any Python object      |Same as for Matrix     |
|                    |of appropriate type. |convertible to 1d      |                       |
|                    |                     |NDarray of appropriate |                       |
|                    |                     |type                   |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
+--------------------+---------------------+-----------------------+-----------------------+
|std vector<int32>   |1-dimensional NDarray|any python iterable    |                       |
|                    |of int32             |                       |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
+--------------------+---------------------+-----------------------+-----------------------+
|std::vector<std::   |list of 1d NDarrays  |Iterable over things   |                       |
|vector<int32>>      |                     |convertible to 1d      |                       |
|                    |                     |NDarrays               |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
+--------------------+---------------------+-----------------------+-----------------------+
|std::               |tuple of ints        |tuple of ints          |                       |
|pair<int32,int32>   |                     |                       |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
+--------------------+---------------------+-----------------------+-----------------------+
|                    | Any Python object   | Any Python object     |Uses repr/eval in text |
|                    |                     |                       |mode and cPickle in    |
|                    |                     |                       |binary mode            |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
|                    |                     |                       |                       |
+--------------------+---------------------+-----------------------+-----------------------+

'''

'''
Created on Jul 31, 2014

@author: chorows
'''


import numpy as np
from kaldi_io_internal import *

if KALDI_BASE_FLOAT()==np.float64:
    RandomAccessBaseFloatMatrixReader = RandomAccessFloat64MatrixReader
    RandomAccessBaseFloatMatrixMapped = RandomAccessFloat64MatrixMapped
    SequentialBaseFloatMatrixReader = SequentialFloat64MatrixReader
    BaseFloatMatrixWriter = Float64MatrixWriter
    
    RandomAccessBaseFloatVectorReader = RandomAccessFloat64VectorReader
    RandomAccessBaseFloatVectorReaderMapped = RandomAccessFloat64VectorReaderMapped
    SequentialBaseFloatVectorReader = SequentialFloat64VectorReader
    BaseFloatVectorWriter = Float64VectorWriter
    
if KALDI_BASE_FLOAT()==np.float32:
    RandomAccessBaseFloatMatrixReader = RandomAccessFloat32MatrixReader
    RandomAccessBaseFloatMatrixMapped = RandomAccessFloat32MatrixMapped
    SequentialBaseFloatMatrixReader = SequentialFloat32MatrixReader
    BaseFloatMatrixWriter = Float32MatrixWriter
    
    RandomAccessBaseFloatVectorReader = RandomAccessFloat32VectorReader
    RandomAccessBaseFloatVectorReaderMapped = RandomAccessFloat32VectorReaderMapped
    SequentialBaseFloatVectorReader = SequentialFloat32VectorReader
    BaseFloatVectorWriter = Float32VectorWriter

def get_io_for_dtype(access, dtype, element=''):
    '''
    Get a writer or reader for the given dtype. eg:
    get_io_for_dtype('Sequential',np.float32,'MatrixReader')
    get_io_for_dtype('float32,'MatrixWriter')
    '''
    if element=='': #assume we want a writer
        access, dtype,element = '',access,dtype
    dtypemap = {np.int32:'Int32',
                np.float32:'Float32',
                np.float64:'Float64',
                'float32':'Float32',
                'float64':'Float64'}
    dtype = dtypemap[dtype]
    return globals()[access + dtype + element] 

class _Transformed(object):
    def __init__(self, reader, transform_function, **kwargs):
        super(_Transformed, self).__init__(**kwargs)
        self.reader=reader
        self.transform_function = transform_function
    
    def __getattr__(self, attr):
        return getattr(self.reader,attr)
    
class TransRA(_Transformed):
    def __init__(self, *args, **kwargs):
        super(TransRA, self).__init__(*args, **kwargs)
    
    def value(self, key):
        return self.transform_function(self.reader.value(key))
    
    def __getitem__(self, key):
        return self.value(key)
    
class TransSeq(_Transformed):
    def __init__(self, *args, **kwargs):
        super(TransSeq, self).__init__(*args, **kwargs)
        
    def next(self):
        return self.transform_function(self.reader.next())

    def _kaldi_value(self):
        return self.transform_function(self.reader._kaldi_value())
    
