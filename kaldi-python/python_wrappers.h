/*
 * python_wrappers.h
 *
 *  Created on: Aug 28, 2014
 *      Author: chorows
 */

#ifndef PYTHON_WRAPPERS_H_
#define PYTHON_WRAPPERS_H_

extern "C" {
#include "Python.h"
#include "numpy/arrayobject.h"
}

#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>

#include <boost/python.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/stl_iterator.hpp>

#include <util/kaldi-io.h>
#include <util/kaldi-table.h>
#include <matrix/kaldi-matrix.h>
#include <matrix/kaldi-vector.h>
#include <util/table-types.h>

namespace kaldi {
//Helper to get proper np type
template <class Real>
int get_np_type() {
  //BOOST_STATIC_ASSERT_MSG(false, "Call one of the explicitly instantiated templates for float or double.");
  KALDI_ERR << "Call one of the explicitly instantiated templates for float or double.";
  return -1;
}

template <>
int get_np_type<double>() {
  return NPY_DOUBLE;
}

template <>
int get_np_type<float>() {
  return NPY_FLOAT;
}

template <>
int get_np_type<kaldi::int32>() {
  return NPY_INT32;
}

template<typename Real>
class NpWrapperMatrix : public kaldi::MatrixBase<Real> {
 public:
  NpWrapperMatrix(PyArrayObject* arr)
      : kaldi::MatrixBase<Real>(),
        arr_(arr) {
    if (PyArray_NDIM(arr_)!=2) {
      PyErr_SetString(PyExc_TypeError, "Can wrap only matrices (2D arrays)");
      boost::python::throw_error_already_set();
    }
    if (PyArray_TYPE(arr)!=get_np_type<Real>()) {
      PyErr_SetString(PyExc_TypeError, "Wrong array dtype");
      boost::python::throw_error_already_set();
    }
    npy_intp* dims = PyArray_DIMS(arr_);
    npy_intp* strides = PyArray_STRIDES(arr_);
    if (strides[1]!=sizeof(Real)) {
      PyErr_SetString(PyExc_TypeError, "Wrong array column stride");
      boost::python::throw_error_already_set();
    }
    Py_INCREF(arr_);
    //why do we have to use this-> in here??
    this->data_ = (Real*)PyArray_DATA(arr);
    this->num_rows_ = dims[0];
    this->num_cols_ = dims[1];
    this->stride_ = strides[0]/sizeof(Real);
  }

  ~NpWrapperMatrix() {
    Py_DECREF(arr_);
  }

 protected:
  PyArrayObject* arr_;
};

template<typename Real>
class NpWrapperVector : public kaldi::VectorBase<Real> {
 public:
  NpWrapperVector(PyArrayObject* arr)
      : kaldi::VectorBase<Real>(),
        arr_(arr) {
    if (PyArray_NDIM(arr_)!=1) {
      PyErr_SetString(PyExc_TypeError, "Can wrap only vectors (1D arrays)");
      boost::python::throw_error_already_set();
    }
    if (PyArray_TYPE(arr)!=get_np_type<Real>()) {
      PyErr_SetString(PyExc_TypeError, "Wrong array dtype");
      boost::python::throw_error_already_set();
    }
    npy_intp* dims = PyArray_DIMS(arr_);
    npy_intp* strides = PyArray_STRIDES(arr_);
    if (strides[0]!=sizeof(Real)) {
      PyErr_SetString(PyExc_TypeError, "Wrong array column stride");
      boost::python::throw_error_already_set();
    }
    Py_INCREF(arr_);
    //why do we have to use this-> in here??
    this->data_ = (Real*)PyArray_DATA(arr);
    this->dim_ = dims[0];
  }

  ~NpWrapperVector() {
    Py_DECREF(arr_);
  }

 protected:
  PyArrayObject* arr_;
};

} //namespace kaldi


#endif /* PYTHON_WRAPPERS_H_ */
