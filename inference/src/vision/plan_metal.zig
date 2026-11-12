//! Weight and buffer helpers the vision projectors' Metal plans share. Each
//! takes the plan as `plan: anytype`, a pointer to a struct with the fields
//! `backend: *metal.Backend`, `view: weights.View`, and `created_bytes:
//! usize`; `created_bytes` counts the device memory the plan created, not
//! the mapped file bytes it wraps in place. Device buffers belong to the
//! backend and are released with it.
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

const Buffer = metal.Buffer;

/// A weight matrix's device buffer and the description the matmul reads.
pub const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };

/// A zeroed device buffer of `len` bytes, counted in `plan.created_bytes`.
pub fn created(plan: anytype, len: usize) !Buffer {
    const buffer = try plan.backend.create(len);
    plan.created_bytes += len;
    return buffer;
}

/// The tensor's mapped bytes, wrapped in place.
pub fn wrapped(plan: anytype, tensor: *const Tensor) !Buffer {
    return plan.backend.wrap(try plan.view.bytes(tensor));
}

/// The matrix tensor's mapped bytes, wrapped in place.
pub fn weight(plan: anytype, tensor: *const Tensor) !Weight {
    const matrix = try plan.view.matrix(tensor);
    return .{ .buffer = try plan.backend.wrap(matrix.bytes), .matrix = matrix };
}

/// A copy of the `rows × columns` matrix (F32, F16, or BF16) with each row
/// zero-padded to `padded` columns, so the matmul tile reads no row past
/// its end; `error.InvalidShape` when the tensor is not that size.
pub fn paddedDown(plan: anytype, tensor: *const Tensor, rows: usize, columns: usize, padded: usize) !Weight {
    const down = try plan.view.matrix(tensor);
    const element: usize = if (down.encoding == 0) 4 else 2;
    if (down.bytes.len != rows * columns * element) return error.InvalidShape;
    const buffer = try created(plan, rows * padded * element);
    const host = buffer.host[0 .. rows * padded * element];
    for (0..rows) |r| @memcpy(host[r * padded * element ..][0 .. columns * element], down.bytes[r * columns * element ..][0 .. columns * element]);
    return .{ .buffer = buffer, .matrix = .{ .encoding = down.encoding, .rows = rows, .columns = padded, .bytes = host } };
}
