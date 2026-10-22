#include "dmm/cannon.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

std::pair<py::array_t<float>, dmm::Trace> cannon_traced(const Array& a, const Array& b, dmm::Executor& executor) {
    auto left = copy_input(a);
    auto right = copy_input(b);
    const auto n = static_cast<py::ssize_t>(left.size);
    py::array_t<float> output({n, n});
    auto* destination = output.mutable_data();
    dmm::Trace trace;
    {
        py::gil_scoped_release release;
        const dmm::Cannon algorithm;
        auto result = algorithm.run(left, right, executor);
        const auto& values = result.product.values;
        std::copy(values.begin(), values.end(), destination);
        trace = std::move(result.trace);
    }
    return {std::move(output), std::move(trace)};
}

py::array_t<float> cannon(const Array& a, const Array& b, dmm::Executor& executor) {
    return cannon_traced(a, b, executor).first;
}
} // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "matmul-style fixed 4x4 Cannon with FP32 inputs, reusable workers and BSP traces.";
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
    module.def("cannon_traced", &cannon_traced,
       py::arg("a").noconvert(), py::arg("b").noconvert(),
       py::arg("executor"));

    py::class_<dmm::MessageRecord>(module, "MessageRecord")
        .def_readonly("source", &dmm::MessageRecord::source)
        .def_readonly("destination", &dmm::MessageRecord::destination)
        .def_readonly("tag", &dmm::MessageRecord::tag)
        .def_readonly("words", &dmm::MessageRecord::words);
    py::class_<dmm::SuperstepRecord>(module, "SuperstepRecord")
        .def_readonly("label", &dmm::SuperstepRecord::label)
        .def_readonly("flops", &dmm::SuperstepRecord::flops)
        .def_readonly("messages", &dmm::SuperstepRecord::messages);
    py::class_<dmm::Trace>(module, "Trace")
        .def_readonly("nodes", &dmm::Trace::nodes)
        .def_readonly("supersteps", &dmm::Trace::supersteps);
    py::class_<dmm::Counts>(module, "Counts")
        .def_readonly("supersteps", &dmm::Counts::supersteps)
        .def_readonly("total_flops", &dmm::Counts::total_flops)
        .def_readonly("critical_flops", &dmm::Counts::critical_flops)
        .def_readonly("total_words", &dmm::Counts::total_words)
        .def_readonly("critical_words", &dmm::Counts::critical_words)
        .def_readonly("messages", &dmm::Counts::messages);
    module.def("count", py::overload_cast<const dmm::Trace&>(&dmm::count), py::arg("trace"));
    module.def("count", py::overload_cast<const dmm::SuperstepRecord&>(&dmm::count), py::arg("superstep"));
    module.def("bsp_cost", [](const dmm::Counts& counts, double w, double g, double l) {
        return dmm::bsp_cost(counts, {w, g, l});
    }, py::arg("counts"), py::kw_only(), py::arg("w") = 1.0, py::arg("g") = 0.0, py::arg("l") = 0.0);
}
