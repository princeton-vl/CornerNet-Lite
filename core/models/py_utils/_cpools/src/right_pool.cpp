#include <torch/torch.h>

#include <vector>

std::vector<at::Tensor> pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);

    // Get width
    int64_t width = input.size(3);

    output.copy_(input);

    for (int64_t ind = 1; ind < width; ind <<= 1) {
        at::Tensor max_temp = at::slice(output, 3, ind, width); 
        at::Tensor cur_temp = at::slice(output, 3, ind, width);        
        at::Tensor next_temp = at::slice(output, 3, 0, width-ind);
        at::max_out(max_temp, cur_temp, next_temp);
    }

    return { 
        output
    };
}

std::vector<at::Tensor> pool_backward(
    at::Tensor input,
    at::Tensor grad_output
) {
    at::Tensor output = at::zeros_like(input);

    int32_t batch   = input.size(0);
    int32_t channel = input.size(1);
    int32_t height  = input.size(2);
    int32_t width   = input.size(3);

    auto max_val = torch::zeros({batch, channel, height}, at::device(at::kCUDA).dtype(at::kFloat));
    auto max_ind = torch::zeros({batch, channel, height}, at::device(at::kCUDA).dtype(at::kLong));

    auto input_temp = input.select(3, 0);
    max_val.copy_(input_temp);

    max_ind.fill_(0);

    auto output_temp      = output.select(3, 0);
    auto grad_output_temp = grad_output.select(3, 0);
    output_temp.copy_(grad_output_temp);

    auto un_max_ind = max_ind.unsqueeze(3);
    auto gt_mask    = torch::zeros({batch, channel, height}, at::device(at::kCUDA).dtype(at::kByte));
    auto max_temp   = torch::zeros({batch, channel, height}, at::device(at::kCUDA).dtype(at::kFloat));
    for (int32_t ind = 0; ind < width - 1; ++ind) {
        input_temp = input.select(3, ind + 1);
        at::gt_out(gt_mask, input_temp, max_val);

        at::masked_select_out(max_temp, input_temp, gt_mask);
        max_val.masked_scatter_(gt_mask, max_temp);
        max_ind.masked_fill_(gt_mask, ind + 1);

        grad_output_temp = grad_output.select(3, ind + 1).unsqueeze(3);
        output.scatter_add_(3, un_max_ind, grad_output_temp);
    }

    return {
        output
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &pool_forward, "Right Pool Forward",
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "backward", &pool_backward, "Right Pool Backward",
        py::call_guard<py::gil_scoped_release>()     
    );
}
