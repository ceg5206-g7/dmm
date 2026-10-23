#include "dmm/cannon.hpp"
#include "dmm/calibration.hpp"
#include "dmm/sharding.hpp"
#include "dmm/coverage.hpp"
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

std::size_t positive_integer(const py::object& value, const char* name) {
    if (!PyLong_CheckExact(value.ptr()) || PyObject_RichCompareBool(value.ptr(), py::int_(0).ptr(), Py_LE) == 1)
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    const auto number = value.cast<std::size_t>();
    if (number == 0) throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return number;
}

dmm::PlanBuilder builder(const std::string& layout) {
    if (layout == "cannon") return dmm::build_cannon;
    if (layout == "ring" || layout == "2.5d")
        throw std::invalid_argument("Layout " + layout + " is not implemented");
    throw std::invalid_argument("unknown layout: " + layout);
}

std::unique_ptr<dmm::PreparedMatmul> prepare(const Array& a, const Array& b,
        const std::string& layout, const py::object& P, const py::object& c) {
    const auto build = builder(layout);
    const auto nodes = positive_integer(P, "P"), copies = positive_integer(c, "c");
    return std::make_unique<dmm::PreparedMatmul>(copy_input(a), copy_input(b), nodes, copies, build);
}

std::pair<py::array_t<float>, dmm::Trace> matmul_traced(const Array& a, const Array& b, dmm::Executor& executor,
        const std::string& layout, const py::object& P, const py::object& c) {
    auto job = prepare(a, b, layout, P, c);
    const auto n = static_cast<py::ssize_t>(job->size());
    py::array_t<float> output({n, n});
    auto* destination = output.mutable_data();
    dmm::Trace trace;
    {
        py::gil_scoped_release release;
        job->run_into({destination, static_cast<std::size_t>(n * n)}, executor);
        trace = job->trace();
    }
    return {std::move(output), std::move(trace)};
}
} // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Matrix multiplication plans with FP32 inputs, reusable workers and BSP traces.";
    module.def("grid_size", [](const py::object& P, const py::object& c) {
        return dmm::grid_size(positive_integer(P, "P"), positive_integer(c, "c"));
    }, py::arg("P"), py::arg("c") = 1);
    module.def("padded_size", [](const py::object& n, const py::object& P) {
        return dmm::padded_size(positive_integer(n, "size"), positive_integer(P, "P"));
    }, py::arg("size"), py::arg("P"));
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
    py::class_<dmm::Timing>(module, "Timing")
        .def_readonly("total", &dmm::Timing::total)
        .def_readonly("compute", &dmm::Timing::compute)
        .def_readonly("copy", &dmm::Timing::copy)
        .def_readonly("synchronization", &dmm::Timing::synchronization);
    py::class_<dmm::RunTimings>(module, "RunTimings")
        .def_readonly("end_to_end", &dmm::RunTimings::end_to_end)
        .def_readonly("phases", &dmm::RunTimings::phases);
    module.attr("phase_names") = dmm::PHASE_NAMES;
    py::class_<dmm::PreparedMatmul>(module, "PreparedMatmul")
        .def("run_into", [](dmm::PreparedMatmul& plan, py::array_t<float, py::array::c_style> output, dmm::Executor& executor) {
            if (output.ndim() != 2 || output.shape(0) != static_cast<py::ssize_t>(plan.size()) ||
                output.shape(1) != static_cast<py::ssize_t>(plan.size())) throw std::invalid_argument("prepared output must match input shape");
            auto* data = output.mutable_data();
            const auto count = static_cast<std::size_t>(output.size());
            py::gil_scoped_release release;
            return plan.run_into({data, count}, executor);
        }, py::arg("output").noconvert(), py::arg("executor"))
        .def("trace", &dmm::PreparedMatmul::trace, py::call_guard<py::gil_scoped_release>());
    module.def("prepare", &prepare,
        py::arg("a").noconvert(), py::arg("b").noconvert(), py::kw_only(),
        py::arg("layout") = "cannon", py::arg("P") = 16, py::arg("c") = 1);
    module.def("microbenchmark", [](const std::string& kind, const py::object& size,
               const py::object& iterations, const py::object& nodes, dmm::Executor& executor) {
        const auto s = positive_integer(size, "size"), it = positive_integer(iterations, "iterations"),
                   P = positive_integer(nodes, "nodes");
        py::gil_scoped_release release;
        return dmm::microbenchmark(kind, s, it, P, executor);
    }, py::arg("kind"), py::arg("size"), py::arg("iterations"), py::arg("nodes"), py::arg("executor"));
    module.def("matmul", [](const Array& a, const Array& b, dmm::Executor& executor,
               const std::string& layout, const py::object& P, const py::object& c) {
        return matmul_traced(a, b, executor, layout, P, c).first;
    }, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("executor"), py::kw_only(),
       py::arg("layout") = "cannon", py::arg("P") = 16, py::arg("c") = 1);
    module.def("matmul_traced", &matmul_traced,
       py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("executor"), py::kw_only(),
       py::arg("layout") = "cannon", py::arg("P") = 16, py::arg("c") = 1);

    py::class_<dmm::ElementRegion>(module, "ElementRegion")
        .def_readonly("row", &dmm::ElementRegion::row)
        .def_readonly("col", &dmm::ElementRegion::col)
        .def_readonly("rows", &dmm::ElementRegion::rows)
        .def_readonly("cols", &dmm::ElementRegion::cols);
    py::class_<dmm::ProductRegion>(module, "ProductRegion")
        .def_readonly("i", &dmm::ProductRegion::i)
        .def_readonly("k", &dmm::ProductRegion::k)
        .def_readonly("j", &dmm::ProductRegion::j)
        .def_readonly("m", &dmm::ProductRegion::m)
        .def_readonly("n", &dmm::ProductRegion::n)
        .def_readonly("inner", &dmm::ProductRegion::inner);
    py::class_<dmm::ProductRecord>(module, "ProductRecord")
        .def_readonly("node", &dmm::ProductRecord::node)
        .def_readonly("product", &dmm::ProductRecord::product);
    py::class_<dmm::CoverageGrid>(module, "CoverageGrid")
        .def_readonly("row_parts", &dmm::CoverageGrid::row_parts)
        .def_readonly("inner_parts", &dmm::CoverageGrid::inner_parts)
        .def_readonly("col_parts", &dmm::CoverageGrid::col_parts);
    py::class_<dmm::MessageRecord>(module, "MessageRecord")
        .def_readonly("source", &dmm::MessageRecord::source)
        .def_readonly("destination", &dmm::MessageRecord::destination)
        .def_readonly("tag", &dmm::MessageRecord::tag)
        .def_readonly("words", &dmm::MessageRecord::words)
        .def_readonly("region", &dmm::MessageRecord::region);
    py::class_<dmm::SuperstepRecord>(module, "SuperstepRecord")
        .def_readonly("label", &dmm::SuperstepRecord::label)
        .def_readonly("flops", &dmm::SuperstepRecord::flops)
        .def_readonly("messages", &dmm::SuperstepRecord::messages)
        .def_readonly("peak_words", &dmm::SuperstepRecord::peak_words)
        .def_readonly("products", &dmm::SuperstepRecord::products);
    py::class_<dmm::Trace>(module, "Trace")
        .def_readonly("nodes", &dmm::Trace::nodes)
        .def_readonly("supersteps", &dmm::Trace::supersteps)
        .def_readonly("peak_words", &dmm::Trace::peak_words)
        .def_readonly("original_size", &dmm::Trace::original_size)
        .def_readonly("padded_size", &dmm::Trace::padded_size)
        .def_readonly("coverage", &dmm::Trace::coverage);
    py::class_<dmm::Counts>(module, "Counts")
        .def_readonly("supersteps", &dmm::Counts::supersteps)
        .def_readonly("total_flops", &dmm::Counts::total_flops)
        .def_readonly("critical_flops", &dmm::Counts::critical_flops)
        .def_readonly("total_words", &dmm::Counts::total_words)
        .def_readonly("critical_words", &dmm::Counts::critical_words)
        .def_readonly("messages", &dmm::Counts::messages)
        .def_readonly("peak_words", &dmm::Counts::peak_words);
    module.def("count", py::overload_cast<const dmm::Trace&>(&dmm::count), py::arg("trace"));
    module.def("count", py::overload_cast<const dmm::SuperstepRecord&>(&dmm::count), py::arg("superstep"));
    module.def("validate_coverage", &dmm::validate_coverage, py::arg("trace"));
    module.def("count_native", &dmm::count_native, py::arg("trace"));
    module.def("bsp_cost", [](const dmm::Counts& counts, double w, double g, double l) {
        return dmm::bsp_cost(counts, {w, g, l});
    }, py::arg("counts"), py::kw_only(), py::arg("w") = 1.0, py::arg("g") = 0.0, py::arg("l") = 0.0);
}
