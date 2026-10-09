use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions, NearestMode};

use super::{make_interpolate_backward_problem, run_interpolate_backward_test};

const NEAREST_BACKWARD_TOLERANCE: f32 = 0.0;

// Scatter from the forward integer mapping rather than duplicating the kernel's
// inverse-boundary calculation. Small integer gradients make equality exact in
// both f16 and f32; NaN padding makes extra row/column reads fail deterministically.
fn check_nearest_backward_integer_mapping(mode: NearestMode) {
    use cubecl::ir::{ElemType, FloatKind, UIntKind};
    use cubek_interpolate::interpolate_backward;
    use cubek_test_utils::{TestInput, supports_f16_arithmetic};

    use super::super::{build_output_tensor, output_host_f32};

    let client = cubecl::test_device().client();
    for kind in [FloatKind::F32, FloatKind::F16] {
        if kind == FloatKind::F16 && !supports_f16_arithmetic(&client) {
            continue;
        }
        let dtype = ElemType::Float(kind);
        for (h, w, oh, ow, channels) in [
            (112, 88, 96, 64, 4),
            (56, 44, 96, 64, 3),
            (28, 22, 96, 64, 4),
            (120, 144, 48, 64, 4),
            (60, 72, 48, 64, 3),
            (30, 36, 48, 64, 4),
            (13, 9, 9, 13, 3),
            (1, 7, 9, 1, 4),
            (7, 1, 1, 9, 3),
            // Exceeds f16's finite range in the coordinate product.
            (257, 1, 255, 1, 4),
            // Fits in small buffers, but the boundary product exceeds u32.
            (65_537, 1, 65_537, 1, 1),
        ] {
            let needs_u64 = (h as u64 * oh as u64) > u32::MAX as u64
                || (w as u64 * ow as u64) > u32::MAX as u64;
            if needs_u64 && !client.properties().features.supports_type(UIntKind::U64) {
                continue;
            }
            let batches = 2;
            let input_shape = vec![batches, h, w, channels];
            let padded_shape = vec![batches, oh + 1, ow + 1, channels];
            let mut grad_data = vec![f32::NAN; padded_shape.iter().product()];
            let mut expected = vec![0.0f32; input_shape.iter().product()];
            let source = |p: usize, input: usize, output: usize| {
                let (p, input, output) = (p as u64, input as u64, output as u64);
                match mode {
                    NearestMode::Floor => (p * input / output) as usize,
                    NearestMode::Exact => ((2 * p + 1) * input / (2 * output)) as usize,
                }
            };
            for b in 0..batches {
                for y in 0..oh {
                    for x in 0..ow {
                        for c in 0..channels {
                            let value = ((b + 3 * y + x + c) % 5) as f32 - 2.0;
                            grad_data[((b * (oh + 1) + y) * (ow + 1) + x) * channels + c] = value;
                            let sy = source(y, h, oh);
                            let sx = source(x, w, ow);
                            expected[((b * h + sy) * w + sx) * channels + c] += value;
                        }
                    }
                }
            }
            let mut grad = TestInput::builder(client.clone(), padded_shape)
                .dtype(dtype)
                .custom(grad_data)
                .generate_without_host_data()
                .binding();
            // Preserve the physical strides, including the guard rows/columns.
            grad.shape = vec![batches, oh, ow, channels].into();
            let input = build_output_tensor(&client, input_shape.clone(), dtype);
            let output = build_output_tensor(&client, input_shape, dtype);
            interpolate_backward(
                &client,
                input.binding(),
                grad,
                output.clone().binding(),
                InterpolateOptions::new(InterpolateMode::Nearest(mode)),
                dtype,
            )
            .unwrap();
            let actual = output_host_f32(&client, output);
            for (index, expected) in expected.into_iter().enumerate() {
                assert_eq!(
                    actual.data.get_f32(index),
                    expected,
                    "{mode:?} {kind:?}: [{h}, {w}] -> [{oh}, {ow}], channels={channels}, index={index}"
                );
            }
        }
    }
}

#[test]
fn test_interpolate_nearest_backward_integer_mapping() {
    check_nearest_backward_integer_mapping(NearestMode::Floor);
}

#[test]
fn test_interpolate_nearest_exact_backward_integer_mapping() {
    check_nearest_backward_integer_mapping(NearestMode::Exact);
}

#[test]
fn test_interpolate_nearest_backward_identity() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 4, 4, 16],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
    );
    run_interpolate_backward_test(client, 5678, -1.0, 1.0, problem, NEAREST_BACKWARD_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_exact_backward_identity() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 4, 4, 16],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact)),
    );
    run_interpolate_backward_test(client, 5678, -1.0, 1.0, problem, NEAREST_BACKWARD_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_backward_upsample() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 10, 10, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
    );
    run_interpolate_backward_test(
        client,
        1234,
        -10.0,
        10.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_exact_backward_upsample() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 10, 10, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact)),
    );
    run_interpolate_backward_test(
        client,
        1234,
        -10.0,
        10.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_backward_downsample() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 2, 2, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
    );
    run_interpolate_backward_test(
        client,
        91011,
        -100.0,
        100.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_exact_backward_downsample() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 2, 2, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact)),
    );
    run_interpolate_backward_test(
        client,
        91011,
        -100.0,
        100.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_backward_resize() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 8, 16, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
    );
    run_interpolate_backward_test(client, 25, -1.0, 1.0, problem, NEAREST_BACKWARD_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_exact_backward_resize() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 8, 16, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact)),
    );
    run_interpolate_backward_test(client, 25, -1.0, 1.0, problem, NEAREST_BACKWARD_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_backward_without_align_corners() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 16, 16, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor))
            .with_align_corners(false),
    );
    run_interpolate_backward_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_exact_backward_without_align_corners() {
    let client = cubecl::test_device().client();
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 16, 16, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact))
            .with_align_corners(false),
    );
    run_interpolate_backward_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}
