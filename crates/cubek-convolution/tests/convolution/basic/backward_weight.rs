use cubecl::{
    ir::AddressType,
    prelude::*,
    std::tensor::layout::{Layout, LayoutExpand},
    zspace::shape,
};
use cubek_convolution::{
    ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
    components::{
        ConvolutionOperation, ConvolutionProblem, Dimensionality,
        global::layout::{OutLayout, OutLayoutLaunch, WeightLayout, WeightLayoutLaunch},
    },
    definition::{BackwardWeightBlueprint, ConvBlueprint},
    launch_ref,
};
use cubek_matmul::definition::AccumulatorOperand;
use cubek_matmul::multi_level::{
    components::global::memory::{Transpose, TransposeLaunch},
    components::tile::TileMatmulKind,
    definition::{BatchMatmulBlueprint, Blueprint, TilingScheme},
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, launch_and_capture_outcome,
};

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes,
};

/// Probe the logical bounds without reading outside either tensor. Robust buffer access
/// on some backends could otherwise hide a missing out-gradient bounds check.
#[cube(launch)]
fn probe_transposed_bounds(
    out_grad: Transpose<OutLayout>,
    weight_grad: Transpose<WeightLayout>,
    result: &mut [u32],
) {
    if ABSOLUTE_POS == 0 {
        let (_, m, k) = out_grad.shape();
        let (_, _, n) = weight_grad.shape();
        result[0] = u32::cast_from(out_grad.is_in_bounds((0, m - 1, k - 1)));
        result[1] = u32::cast_from(out_grad.is_in_bounds((0, m, k - 1)));
        result[2] = u32::cast_from(out_grad.is_in_bounds((0, m - 1, k)));
        result[3] = u32::cast_from(out_grad.is_in_bounds((0, m, k)));
        result[4] = u32::cast_from(weight_grad.is_in_bounds((0, m - 1, n - 1)));
        result[5] = u32::cast_from(weight_grad.is_in_bounds((0, m, n - 1)));
        result[6] = u32::cast_from(weight_grad.is_in_bounds((0, m - 1, n)));
        result[7] = u32::cast_from(weight_grad.is_in_bounds((0, m, n)));
    }
}

/// End padding is encoded by the out-gradient extent. With input length 8, a three-tap kernel,
/// no beginning padding, and end padding 2, the output length is 8. The three weight taps then
/// overlap 8, 7, and 6 real input positions respectively.
#[test]
fn backward_weight_supports_end_only_padding() {
    run_backward_weight_case(
        8,
        16,
        16,
        default_tiling_scheme(),
        ConvAlgorithm::SimpleSyncCyclic,
    );
}

#[test]
fn backward_weight_partial_output_tile() {
    let mut tiling = default_tiling_scheme();
    tiling.partition_size.m = (32 / tiling.tile_size.m) as u8;
    tiling.partition_size.n = (32 / tiling.tile_size.n) as u8;
    // M = 64 and K = 16 are aligned, but N = 3 * 16 is smaller than the 64-wide stage.
    run_backward_weight_case(16, 16, 64, tiling, ConvAlgorithm::SimpleSyncCyclic);
}

#[test]
fn backward_weight_partial_reduction_tile() {
    // M and N are aligned to the default stage, while K = 17 requires a read bound.
    run_backward_weight_case(
        17,
        32,
        32,
        default_tiling_scheme(),
        ConvAlgorithm::SimpleSyncCyclic,
    );
}

/// Exercise all combinations of aligned and partial M, N and K tiles. In backward weight,
/// the out-gradient and weight-gradient layouts transpose different pairs of these axes.
#[cfg(feature = "heavy")]
#[test]
fn backward_weight_cyclic_transposed_bounds() {
    run_transposed_bounds_cases(ConvAlgorithm::SimpleSyncCyclic);
}

#[cfg(feature = "heavy")]
#[test]
fn backward_weight_tilewise_transposed_bounds() {
    run_transposed_bounds_cases(ConvAlgorithm::SimpleSyncTilewise);
}

#[cfg(feature = "heavy")]
fn run_transposed_bounds_cases(algorithm: ConvAlgorithm) {
    let mut tiling = default_tiling_scheme();
    // A 128x128 stage makes N = 3 * 64 partial, while M = 128 is aligned.
    // Without transposing the bounds checks, the extra output columns can wrap back
    // onto the first kernel tap and overwrite valid weight gradients.
    tiling.partition_size.m = (64 / tiling.tile_size.m) as u8;
    tiling.partition_size.n = (64 / tiling.tile_size.n) as u8;

    for out_channels in [128, 120] {
        for channels in [128, 64] {
            for out_len in [32, 33] {
                run_backward_weight_case(out_len, channels, out_channels, tiling, algorithm);
            }
        }
    }

    // Also cover physical channel padding, independently of the matmul tile bounds.
    run_backward_weight_case(33, 17, 31, tiling, algorithm);
}

fn run_backward_weight_case(
    out_len: usize,
    channels: usize,
    out_channels: usize,
    tiling: TilingScheme,
    algorithm: ConvAlgorithm,
) {
    let client = cubecl::test_device().client();
    let dtypes = f16_dtypes();

    let batches = 1;
    let in_len = out_len;
    let kernel_len = 3;
    let channel_align = client.properties().io_width() as usize / dtypes.lhs_global.size_bits();
    let padded_channels = channels.next_multiple_of(channel_align);

    let input = TestInput::builder(client.clone(), shape![batches, in_len, channels])
        .dtype(dtypes.lhs_global)
        .custom(vec![1.; batches * in_len * channels])
        .generate_without_host_data();
    let out_grad = TestInput::builder(client.clone(), shape![batches, out_len, out_channels])
        .dtype(dtypes.rhs_global)
        .custom(vec![1.; batches * out_len * out_channels])
        .generate_without_host_data();
    let weight_grad =
        TestInput::builder(client.clone(), shape![out_channels, kernel_len, channels])
            .dtype(dtypes.acc_global)
            .zeros()
            .generate_without_host_data();

    let problem = ConvolutionProblem {
        m: out_channels,
        n: padded_channels * kernel_len,
        k: batches * out_len,
        lhs_strides: input.strides().clone(),
        rhs_strides: out_grad.strides().clone(),
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        kernel_size: vec![kernel_len as u32],
        stride: vec![1],
        padding: vec![0],
        dilation: vec![1],
        batches,
        channels,
        out_channels,
        in_shape: shape![in_len],
        out_shape: shape![out_len],
        padded_channels,
        operation: ConvolutionOperation::BackwardWeight,
        dimensionality: Dimensionality::Dim1,
        global_dtypes: dtypes.as_global_elems(),
        address_type: AddressType::U32,
    };
    let matmul = BatchMatmulBlueprint::builder(
        TileMatmulKind::Cmma,
        tiling,
        client.properties().hardware.plane_size_max,
        &problem.as_matmul_problem(AccumulatorOperand::Absent),
    )
    .shared_swizzle(default_swizzle())
    .partition_buffering(default_partition_buffering())
    .build();

    let bounds = client.empty(8 * core::mem::size_of::<u32>());
    probe_transposed_bounds::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        TransposeLaunch::new(OutLayoutLaunch::from_args(
            &problem,
            matmul.lhs_global_layout_config(),
        )),
        TransposeLaunch::new(WeightLayoutLaunch::from_args(
            &problem,
            matmul.out_global_layout_config(),
        )),
        // The buffer contains exactly eight u32 values.
        unsafe { BufferArg::from_raw_parts(bounds.clone(), 8) },
    );
    let actual_bounds = client.read_one(bounds).unwrap();
    let actual_bounds: &[u32] = bytemuck::cast_slice(&actual_bounds);
    let m = !matmul.check_m_bounds;
    let n = !matmul.check_n_bounds;
    let k = !matmul.check_k_bounds;
    assert_eq!(
        actual_bounds,
        [true, m, k, m && k, true, m, n, m && n].map(u32::from),
        "transposed bounds for M={}, N={}, K={}",
        problem.m,
        problem.n,
        problem.k,
    );

    let inputs = ConvolutionInputs::BackwardWeight {
        input: InputBinding::new(input.binding(), dtypes.lhs_global),
        out_grad: InputBinding::new(out_grad.binding(), dtypes.rhs_global),
        weight_grad: weight_grad.clone().binding(),
    };
    let args = ConvolutionArgs::<1> {
        stride: [1],
        padding: [0],
        dilation: [1],
    };
    let strategy = Strategy::Forced {
        algorithm,
        blueprint: ConvBlueprint::BackwardWeight(BackwardWeightBlueprint {
            matmul,
            dimensionality: Dimensionality::Dim1,
        }),
    };

    let outcome = launch_and_capture_outcome(&client, &[&weight_grad.handle], |client| {
        match launch_ref(&strategy, client, inputs, args, dtypes) {
            Ok(()) => ExecutionOutcome::Executed,
            Err(error) => ExecutionOutcome::CompileError(format!("{error:?}")),
        }
    });

    if let ExecutionOutcome::CompileError(error) = outcome {
        TestOutcome::CompileError(error).enforce();
        return;
    }

    let actual = HostData::from_tensor_handle(&client, weight_grad, HostDataType::F32);
    let expected_per_tap = [out_len as f32, (out_len - 1) as f32, (out_len - 2) as f32];

    for out_channel in 0..out_channels {
        for (tap, expected) in expected_per_tap.into_iter().enumerate() {
            for channel in 0..channels {
                assert_eq!(
                    actual.get_f32(&[out_channel, tap, channel]),
                    expected,
                    "{algorithm:?}: input length {in_len}, channels {channels}, output channels \
                     {out_channels}, weight [{out_channel}, {tap}, {channel}]"
                );
            }
        }
    }
}
