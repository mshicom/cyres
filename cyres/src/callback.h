#include <Python.h>
using ceres::IterationCallback;
using ceres::CallbackReturnType;
using ceres::IterationSummary;

extern "C" CallbackReturnType cy_callback(PyObject* obj, PyObject* summary);

extern "C" PyObject* cy_warpSummary(const IterationSummary& summary);

class PythonCallback : public IterationCallback {
 public:
  explicit PythonCallback(PyObject* self_): self_(self_) {
    Py_XINCREF(self_);
  }

  ~PythonCallback() {Py_XDECREF(self_);}

  CallbackReturnType operator()(const IterationSummary& summary) {
   return cy_callback(self_, cy_warpSummary(summary));
  }


 private:
  PyObject* self_;
};
