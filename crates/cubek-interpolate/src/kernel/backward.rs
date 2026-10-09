use crate::definition::NearestMode;
use cubecl::prelude::*;
use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    ir::UIntKind,
    num_traits::Zero,
    prelude::TensorBinding,
    std::FastDivmod,
    std::tensor::layout::{
        linear::{LinearLayout, LinearLayoutLaunch, LinearViewLayoutLaunch},
        *,
    },
    tensor_vector_size_parallel,
};

pub(crate) fn interpolate_nearest_backward_launch(
    client: &Client,
    out_grad: TensorBinding,
    input_grad: TensorBinding,
    nearest_mode: NearestMode,
    dtype: ElemType,
) -> Result<(), crate::definition::InterpolateError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &out_grad.shape,
        &out_grad.strides,
        out_grad.shape.len() - 1,
    );
    let input_grad_shape = shape_divmod(&input_grad);
    let input_grad_layout = linear_layout(&input_grad, vector_size);
    let working_units = input_grad.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    let address_type = out_grad
        .required_address_type(dtype.size())
        .max(input_grad.required_address_type(dtype.size()));
    let index_dtype = boundary_dtype(&input_grad.shape[1..3], &out_grad.shape[1..3]);

    unsafe {
        execute_interpolate_nearest_backward::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            out_grad.into_tensor_arg(),
            input_grad.clone().into_tensor_arg(),
            input_grad_shape,
            input_grad_layout,
            nearest_mode,
            dtype,
            index_dtype,
        )
    };
    Ok(())
}

// Small tensors can still have coordinate products exceeding u32 (for example,
// a 65537 x 1 identity resize). Shape validation limits each extent to i32::MAX,
// so their product fits in u64. Ordinary image sizes retain 32-bit indexing.
fn boundary_dtype(input_size: &[usize], output_size: &[usize]) -> ElemType {
    if input_size.iter().zip(output_size).any(|(input, output)| {
        input
            .checked_mul(*output)
            .is_none_or(|product| product > u32::MAX as usize)
    }) {
        ElemType::UInt(UIntKind::U64)
    } else {
        ElemType::UInt(UIntKind::U32)
    }
}

fn shape_divmod(binding: &TensorBinding) -> SequenceArg<FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

fn linear_layout(binding: &TensorBinding, vector_size: usize) -> LinearLayoutLaunch {
    LinearLayoutLaunch::from_shape_strides(
        binding.shape.clone(),
        binding.strides.clone(),
        Type::new(ElemType::UInt(UIntKind::U32)).with_vector_size(vector_size),
        LinearViewLayoutLaunch::new(),
    )
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn execute_interpolate_nearest_backward<F: Float, I: Int, N: Size>(
    grad: &Tensor<Vector<F, N>>,
    input_grad: &mut Tensor<Vector<F, N>>,
    input_grad_shape: Sequence<FastDivmod<usize>>,
    input_grad_layout: LinearLayout,
    #[comptime] nearest_mode: NearestMode,
    #[define(F)] _dtype: ElemType,
    #[define(I)] _index_dtype: ElemType,
) {
    if ABSOLUTE_POS >= input_grad.len() {
        terminate!();
    }

    let vector_size = grad.vector_size();
    let input_grad_idx = input_grad_layout.to_source_pos(ABSOLUTE_POS);

    let out_h = input_grad.shape(1);
    let out_w = input_grad.shape(2);
    let grad_h = grad.shape(1);
    let grad_w = grad.shape(2);

    let (rem, c) = input_grad_shape[3].div_mod(ABSOLUTE_POS * vector_size);
    let (rem, out_x) = input_grad_shape[2].div_mod(rem);
    let (b, out_y) = input_grad_shape[1].div_mod(rem);

    let grad_y_start = start_index::<I>(out_y, grad_h, out_h, nearest_mode);
    let grad_y_end = end_index::<I>(out_y, grad_h, out_h, nearest_mode);
    let grad_x_start = start_index::<I>(out_x, grad_w, out_w, nearest_mode);
    let grad_x_end = end_index::<I>(out_x, grad_w, out_w, nearest_mode);

    let index_grad_base = b * grad.stride(0) + c * grad.stride(3);

    let mut sum = Vector::zero();

    for grad_y in grad_y_start..grad_y_end {
        for grad_x in grad_x_start..grad_x_end {
            let index_grad = index_grad_base + grad_y * grad.stride(1) + grad_x * grad.stride(2);

            sum += grad[index_grad / vector_size];
        }
    }

    input_grad[input_grad_idx] = sum;
}

#[cube]
fn start_index<I: Int>(
    input_index: usize,
    output_size: usize,
    input_size: usize,
    #[comptime] nearest_mode: NearestMode,
) -> usize {
    // Invert floor(j * input_size / output_size), or the half-pixel mapping
    // floor((j + 0.5) * input_size / output_size) for Exact. Floating division
    // followed by ceil can overshoot even in f32, shifting interior boundaries
    // and reading past grad at the last boundary. Do not use the tensor dtype
    // for coordinates: f16 also rounds/overflows these products.
    let numerator = I::cast_from(input_index) * I::cast_from(output_size);
    let divisor = I::cast_from(input_size);
    let quotient = numerator / divisor;
    let remainder = numerator - quotient * divisor;
    match nearest_mode {
        NearestMode::Floor => {
            usize::cast_from(quotient + I::cast_from(remainder != I::from_int(0)))
        }
        // ceil(n / d - 0.5), clamped to zero, without doubling n or subtracting
        // from an unsigned value. This also preserves exact half-pixel ties.
        NearestMode::Exact => {
            usize::cast_from(quotient + I::cast_from(remainder > divisor / I::from_int(2)))
        }
    }
}

#[cube]
fn end_index<I: Int>(
    input_index: usize,
    output_size: usize,
    input_size: usize,
    #[comptime] nearest_mode: NearestMode,
) -> usize {
    start_index::<I>(input_index + 1, output_size, input_size, nearest_mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_products_select_sufficient_integer_width() {
        assert_eq!(
            boundary_dtype(&[120, 144], &[48, 64]),
            ElemType::UInt(UIntKind::U32)
        );
        assert_eq!(
            boundary_dtype(&[65_535, 1], &[65_537, 1]),
            ElemType::UInt(UIntKind::U32)
        );
        assert_eq!(
            boundary_dtype(&[65_536, 1], &[65_536, 1]),
            ElemType::UInt(UIntKind::U64)
        );
        assert_eq!(
            boundary_dtype(&[1, 65_537], &[1, 65_537]),
            ElemType::UInt(UIntKind::U64)
        );
        let max = i32::MAX as usize;
        assert_eq!(
            boundary_dtype(&[max, 1], &[max, 1]),
            ElemType::UInt(UIntKind::U64)
        );
    }
}
