#include "dmm/cannon.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

namespace {
using Array = py::array_t<float, 0>;

dmm::Matrix<float> copy_input(const Array& array) {
    if (array.ndim() != 2 || array.shape(0) == 0 || array.shape(0) != array.shape(1))
        throw std::invalid_argument("inputs must be non-empty square float32 arrays");
    dmm::Matrix<float> matrix(static_cast<std::size_t>(array.shape(0)));
    const auto view = array.unchecked<2>();
    for (py::ssize_t i = 0; i < array.shape(0); ++i)
        for (py::ssize_t j = 0; j < array.shape(1); ++j) {
            const auto value = view(i, j);
            if (!std::isfinite(value)) throw std::invalid_argument("inputs must contain finite values");
            matrix.values[static_cast<std::size_t>(i) * matrix.size + static_cast<std::size_t>(j)] = value;
        }
    return matrix;
}

std::size_t worker_count(const py::object& value) {
    if (!PyLong_CheckExact(value.ptr()))
        throw std::invalid_argument("workers must be an integer");
    const auto count = value.cast<py::ssize_t>();
    if (count < 0 || count > std::max(1u, std::thread::hardware_concurrency()))
        throw std::invalid_argument("workers must be between zero and the available logical CPU count");
    return static_cast<std::size_t>(count);
}

py::array_t<float> cannon(const Array& a, const Array& b, dmm::Executor& executor) {
    auto left = copy_input(a);
    auto right = copy_input(b);
    const auto n = static_cast<py::ssize_t>(left.size);
    py::array_t<float> output({n, n});
    auto* destination = output.mutable_data();
    {
        py::gil_scoped_release release;
        const dmm::Cannon algorithm;
        const auto result = algorithm.matmul(left, right, executor);
        const auto& values = result.values;
        std::copy(values.begin(), values.end(), destination);
    }
    return output;
}
} // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "matmul-style fixed 4x4 Cannon with FP32 inputs and reusable workers.";
    module.attr("GRID") = dmm::GRID;
    module.attr("compiler") = __VERSION__;
    module.attr("build_type") = DMM_BUILD_TYPE;
    py::class_<dmm::Executor>(module, "Executor");
    py::class_<dmm::SerialExecutor, dmm::Executor>(module, "SerialExecutor")
        .def(py::init<>())
        .def("__enter__", [](dmm::SerialExecutor& executor) -> dmm::SerialExecutor& {
            return executor;
        }, py::return_value_policy::reference_internal)
        .def("__exit__", [](dmm::SerialExecutor&, const py::object&, const py::object&, const py::object&) {
            return false;
        });
    py::class_<dmm::ThreadExecutor, dmm::Executor>(module, "ThreadExecutor")
        .def(py::init([](const py::object& workers) {
            return std::make_unique<dmm::ThreadExecutor>(worker_count(workers));
        }), py::arg("workers"))
        .def("__enter__", &dmm::ThreadExecutor::enter,
             py::return_value_policy::reference_internal, py::call_guard<py::gil_scoped_release>())
        .def("__exit__", [](dmm::ThreadExecutor& executor, const py::object&, const py::object&, const py::object&) {
            executor.close();
            return false;
        }, py::call_guard<py::gil_scoped_release>())
        .def("close", &dmm::ThreadExecutor::close, py::call_guard<py::gil_scoped_release>());
    module.def("cannon", &cannon,
       py::arg("a").noconvert(), py::arg("b").noconvert(),
       py::arg("executor"));
}
