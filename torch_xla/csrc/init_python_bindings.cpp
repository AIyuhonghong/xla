#include "init_python_bindings.h"

#include "module.h"
#include "passes/eval_static_size.h"
#include "passes/insert_explicit_expand.h"
#include "passes/replace_untraced_operators.h"
#include "passes/set_mat_mul_output_shape.h"
#include "passes/threshold_backward_peephole.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch_util.h"

namespace torch {
namespace jit {

namespace {

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

void InitXlaModuleBindings(py::module m) {
  py::class_<XlaModule, std::shared_ptr<XlaModule>>(m, "XlaModule")
      .def(py::init([](const std::shared_ptr<script::Module> module,
                       bool use_full_conv_precision, bool differentiate) {
             return std::make_shared<XlaModule>(module, use_full_conv_precision,
                                                differentiate);
           }),
           py::arg("module"), py::arg("use_full_conv_precision") = false,
           py::arg("differentiate") = true)
      .def("__call__",
           [](XlaModule& xla_module, py::args args) -> py::object {
             auto inputs = XlaCreateTensorList(args);
             XlaModule::TensorBatchVector outputs;
             {
               NoGilSection nogil;
               outputs = xla_module.forward(inputs);
             }
             return XlaPackTensorList(outputs);
           })
      .def("backward",
           [](XlaModule& xla_module, py::args args) {
             auto inputs = XlaCreateTensorList(args);
             NoGilSection nogil;
             xla_module.backward(inputs);
           })
      .def("set_inputs_gardients",
           [](XlaModule& xla_module, const py::list& gradient_list) {
             std::vector<at::Tensor> gradients;
             for (auto& gradient : gradient_list) {
               gradients.push_back(gradient.cast<at::Tensor>());
             }
             xla_module.SetInputGradientsForFusion(std::move(gradients));
           })
      .def("parameters",
           [](XlaModule& xla_module) { return xla_module.parameters(); })
      .def("parameters_buffers", [](XlaModule& xla_module) {
        return xla_module.parameters_buffers();
      });
  m.def("_xla_mul_add_multi",
        [](const double scale_dest,
           const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
           const double alpha,
           const std::vector<std::shared_ptr<XLATensor>>& source_tuple) {
          XLATensor::MulAddMulti(scale_dest, dest_tuple, alpha, source_tuple);
        });
  m.def("_xla_zero_multi",
        [](const std::vector<std::shared_ptr<XLATensor>>& dest_tuple) {
          XLATensor::ZeroMulti(dest_tuple);
        });
  m.def("_xla_sync_multi",
        [](const std::vector<std::shared_ptr<XLATensor>>& tensors) {
          NoGilSection nogil;
          XLATensor::ApplyPendingGraph(tensors);
        });
  m.def("_xla_to_tensors",
        [](const std::vector<std::shared_ptr<XLATensor>>& tensors) {
          std::vector<at::Tensor> result;
          {
            NoGilSection nogil;
            result = XLATensor::GetTensors(tensors);
          }
          return result;
        });
  m.def("_xla_create_tensors",
        [](const std::vector<autograd::Variable>& tensors,
           const std::vector<std::string>& devices) {
          std::vector<std::shared_ptr<XLATensor>> result;
          {
            NoGilSection nogil;
            result = XLATensor::CreateTensors(tensors, devices);
          }
          return result;
        });
  m.def("_xla_metrics_report",
        []() { return xla::metrics::CreateMetricReport(); });
}

void InitXlaPassesBindings(py::module m) {
  m.def("_jit_pass_eval_static_size", EvalStaticSize);
  m.def("_jit_pass_replace_untraced_operators", ReplaceUntracedOperators);
  m.def("_jit_pass_threshold_backward_peephole", ThresholdBackwardPeephole);
  m.def("_jit_pass_set_mat_mul_output_shape", SetMatMulOutputShape);
  m.def("_jit_pass_insert_explicit_expand", InsertExplicitExpand);
}

void InitXlaTensorBindings(py::module m) {
  py::class_<XLATensor, std::shared_ptr<XLATensor>>(m, "XLATensor")
      .def(py::init([](autograd::Variable tensor, const std::string& device) {
             return XLATensor::Create(tensor,
                                      XLATensor::DeviceFromString(device));
           }),
           py::arg("tensor"), py::arg("device") = "")
      .def("to_tensor", [](XLATensor& s) { return s.toTensor(); })
      .def("size", [](const XLATensor& s) { return s.Size(); })
      .def("device",
           [](const XLATensor& s) { return s.GetDevice().ToString(); })
      .def("__add__", [](std::shared_ptr<XLATensor> self,
                         XLATensor& other) { return self->add(other, 1.0); })
      .def("add", [](std::shared_ptr<XLATensor> self, double alpha,
                     XLATensor& other) { return self->add(other, alpha); })
      .def("add_",
           [](std::shared_ptr<XLATensor> self, double alpha, XLATensor& other) {
             self->add_(other, alpha);
             return self;
           })
      .def("add_",
           [](std::shared_ptr<XLATensor> self, XLATensor& other) {
             self->add_(other, 1.);
             return self;
           })
      .def("__mul__",
           [](std::shared_ptr<XLATensor> self, XLATensor& other) {
             return self->mul(other);
           },
           py::arg("other"))
      .def("__mul__", [](std::shared_ptr<XLATensor> self,
                         double other) { return self->mul(other); })
      .def("mul",
           [](std::shared_ptr<XLATensor> self, XLATensor& other) {
             return self->mul(other);
           },
           py::arg("other"))
      .def("mul", [](std::shared_ptr<XLATensor> self,
                     double other) { return self->mul(other); })
      .def("mul_",
           [](std::shared_ptr<XLATensor> self, XLATensor& other) {
             self->mul_(other);
             return self;
           },
           py::arg("other"))
      .def("mul_",
           [](std::shared_ptr<XLATensor> self, double other) {
             self->mul_(other);
             return self;
           })
      .def("__div__",
           [](std::shared_ptr<XLATensor> self, XLATensor& other) {
             return self->div(other);
           },
           py::arg("other"))
      .def("__div__", [](std::shared_ptr<XLATensor> self,
                         double other) { return self->div(other); })
      .def("__truediv__",
           [](std::shared_ptr<XLATensor> self, XLATensor& other) {
             return self->div(other);
           },
           py::arg("other"))
      .def("__truediv__", [](std::shared_ptr<XLATensor> self,
                             double other) { return self->div(other); })
      .def("cross_replica_sum",
           [](std::shared_ptr<XLATensor> self, const py::list& groups) {
             std::vector<std::vector<xla::int64>> crs_groups;
             for (auto& group : groups) {
               crs_groups.emplace_back();
               for (auto& replica_id : group.cast<py::list>()) {
                 crs_groups.back().push_back(replica_id.cast<xla::int64>());
               }
             }
             return self->cross_replica_sum(crs_groups);
           })
      .def("zero_",
           [](std::shared_ptr<XLATensor> self) {
             self->zero_();
             return self;
           })
      .def("detach_",
           [](std::shared_ptr<XLATensor> self) {
             self->detach_();
             return self;
           })
      .def_property_readonly(
          "data",
          [](std::shared_ptr<XLATensor> self) {
            return py::cast<std::shared_ptr<XLATensor>>(self->Clone());
          })
      .def_property_readonly(
          "dtype",
          [](std::shared_ptr<XLATensor> self) {
            return py::cast<py::object>(
                torch::autograd::utils::wrap(torch::getDtype(self->dtype())));
          })
      .def_property_readonly("is_leaf", [](const XLATensor&) { return true; })
      .def_property_readonly(
          "grad",
          [](XLATensor& m) -> py::object {
            if (m.grad() == nullptr) {
              return py::none();
            } else {
              return py::cast<std::shared_ptr<XLATensor>>(m.grad());
            }
          })
      .def("__repr__", [](XLATensor& m) {
        std::ostringstream s;
        s << m.toTensor();
        return s.str();
      });
}

}  // namespace

void InitXlaBindings(py::module m) {
  InitXlaModuleBindings(m);
  InitXlaPassesBindings(m);
  InitXlaTensorBindings(m);
}

}  // namespace jit
}  // namespace torch

PYBIND11_MODULE(_XLAC, m) { torch::jit::InitXlaBindings(m); }
