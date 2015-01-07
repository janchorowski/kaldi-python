/*
 * bp_converters.h
 *
 *  Created on: Aug 28, 2014
 *      Author: chorows
 */

#ifndef BP_CONVERTERS_H_
#define BP_CONVERTERS_H_

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>

#include <boost/python.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/stl_iterator.hpp>



namespace kaldi {
//
// Code transformend from http://code.activestate.com/lists/python-cplusplus-sig/16463/ and
// http://misspent.wordpress.com/2009/09/27/how-to-write-boost-python-converters/
//
template<typename T>
struct VectorToListBPConverter {

  static PyObject* convert(std::vector<T> const& vec) {
    boost::python::list l;

    for (size_t i = 0; i < vec.size(); i++)
      l.append(vec[i]);
    return boost::python::incref(l.ptr());
  }
};

template<typename T>
struct VectorFromListBPConverter {
  VectorFromListBPConverter() {
    using namespace boost::python;
    using namespace boost::python::converter;
    boost::python::converter::registry::push_back(
        &VectorFromListBPConverter<T>::convertible,
        &VectorFromListBPConverter<T>::construct, type_id<std::vector<T> >());
  }

  // Determine if obj_ptr can be converted in a std::vector<T>
  static void* convertible(PyObject* obj_ptr) {
//    if (!PyIter_Check(obj_ptr)) {
//      return 0;
//    }
    return obj_ptr;
  }

  // Convert obj_ptr into a std::vector<T>
  static void construct(
      PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data) {

    boost::python::object o = boost::python::object(boost::python::handle<>(boost::python::borrowed(obj_ptr)));
    boost::python::stl_input_iterator<T> begin(o);
    boost::python::stl_input_iterator<T> end;

    // Grab pointer to memory into which to construct the new std::vector<T>
    void* storage = ((boost::python::converter::rvalue_from_python_storage<
        std::vector<T> >*) data)->storage.bytes;

    // in-place construct the new std::vector<T> using the character data
    // extraced from the python object
    std::vector<T>& v = *(new (storage) std::vector<T>());

    v.insert(v.end(), begin, end);

    // Stash the memory chunk pointer for later use by boost.python
    data->convertible = storage;
  }
};

template<typename M>
struct MapFromDictBPConverter {
  MapFromDictBPConverter() {
    boost::python::converter::registry::push_back(
        &MapFromDictBPConverter<M>::convertible,
        &MapFromDictBPConverter<M>::construct, boost::python::type_id<M>());
  }

  // Determine if obj_ptr can be converted in a std::vector<T>
  static void* convertible(PyObject* obj_ptr) {
    if (!PyDict_Check(obj_ptr)) {
      return 0;
    }
    return obj_ptr;
  }

  // Convert obj_ptr into a std::vector<T>
  static void construct(
      PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data) {

    boost::python::dict obj(boost::python::handle<>(boost::python::borrowed(obj_ptr)));
    boost::python::list keys = obj.keys();

    // Grab pointer to memory into which to construct the new std::vector<T>
    void* storage = ((boost::python::converter::rvalue_from_python_storage< M >*) data)->storage.bytes;

    M& map = *(new (storage) M());

    boost::python::stl_input_iterator<typename M::key_type> begin(keys);
    boost::python::stl_input_iterator<typename M::key_type> end;

    for (;begin!=end; ++begin) {
      const typename M::key_type& k = *begin;
      const typename M::mapped_type& v = boost::python::extract<typename M::mapped_type>(obj[k]);
      map[k] = v;
    }

    // Stash the memory chunk pointer for later use by boost.python
    data->convertible = storage;
  }
};


template<typename T1, typename T2>
struct PairToTupleBPConverter {

  static PyObject* convert(std::pair<T1,T2> const& p) {
    return boost::python::incref(boost::python::make_tuple(p.first, p.second).ptr());
  }
};

template<typename T1, typename T2>
struct PairFromTupleBPConverter {
  PairFromTupleBPConverter() {
    boost::python::converter::registry::push_back(
        &PairFromTupleBPConverter<T1, T2>::convertible,
        &PairFromTupleBPConverter<T1, T2>::construct, boost::python::type_id<std::pair<T1,T2> >());
  }

  // Determine if obj_ptr can be converted in a std::vector<T>
  static void* convertible(PyObject* obj_ptr) {
    if (!PyTuple_Check(obj_ptr) || PySequence_Length(obj_ptr)!=2) {
      return 0;
    }
    return obj_ptr;
  }

  // Convert obj_ptr into a std::vector<T>
  static void construct(
      PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data) {

    boost::python::tuple t = boost::python::tuple(boost::python::handle<>(boost::python::borrowed(obj_ptr)));

    // Grab pointer to memory into which to construct the new std::vector<T>
    void* storage = ((boost::python::converter::rvalue_from_python_storage<
        std::pair<T1,T2> >*) data)->storage.bytes;

    // in-place construct the new std::vector<T> using the character data
    // extraced from the python object
    std::pair<T1,T2>& v = *(new (storage) std::pair<T1,T2>());

    v.first=boost::python::extract<T1>(t[0]);
    v.second=boost::python::extract<T2>(t[1]);

    // Stash the memory chunk pointer for later use by boost.python
    data->convertible = storage;
  }
};


}

#endif /* BP_CONVERTERS_H_ */
