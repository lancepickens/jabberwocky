// MetalProcessor.swift – GPU-accelerated image processing via Metal
//
// Uses Apple Silicon's unified memory architecture – CPU-written buffers are
// directly visible to the GPU without any copy, making the pipeline extremely
// efficient on M-series chips.
//
// Pipeline:
//   1. debayer()  – converts a raw Bayer mosaic to an RGBA float32 texture
//   2. stack()    – accumulates registered frames and divides by frame count
//
// Shader source is embedded as a string and compiled at runtime to avoid
// the need for a Metal library bundle (works with Swift Package Manager).

import Foundation
import Metal
import Accelerate

// MARK: - Embedded MSL Shader Source

private let kShaderSource = """
#include <metal_stdlib>
using namespace metal;

// ── Bilinear Bayer demosaicing ─────────────────────────────────────────────
//
// Supports RGGB, BGGR, GRBG, GBRG patterns.
// 'patternOffset' encodes the (col, row) of the Red pixel:
//   RGGB -> (0,0)   BGGR -> (1,1)   GRBG -> (1,0)   GBRG -> (0,1)
//
// Each GPU thread handles one output pixel. On Apple Silicon the GPU can
// process several thousand pixels per clock cycle.

kernel void debayerKernel(
    texture2d<float, access::read>  bayerTex  [[texture(0)]],
    texture2d<float, access::write> rgbaTex   [[texture(1)]],
    constant int2&                  rOffset   [[buffer(0)]],   // position of Red in 2x2 cell
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = bayerTex.get_width();
    const uint H = bayerTex.get_height();
    if (gid.x >= W || gid.y >= H) return;

    int x = int(gid.x);
    int y = int(gid.y);

    // Determine colour at this Bayer position relative to pattern origin
    int cx = (x + 2 - rOffset.x) % 2;   // 0 = red/blue column, 1 = green column
    int cy = (y + 2 - rOffset.y) % 2;   // 0 = red/blue row,    1 = green row

    // Safe-clamped read of a single Bayer sample
    auto bayer = [&](int px, int py) -> float {
        px = clamp(px, 0, int(W) - 1);
        py = clamp(py, 0, int(H) - 1);
        return bayerTex.read(uint2(px, py)).r;
    };

    float r, g, b;

    if (cy == 0 && cx == 0) {
        // ── Red pixel ──────────────────────────────────────────────────────
        r = bayer(x, y);
        g = (bayer(x-1,y) + bayer(x+1,y) + bayer(x,y-1) + bayer(x,y+1)) * 0.25f;
        b = (bayer(x-1,y-1) + bayer(x+1,y-1) + bayer(x-1,y+1) + bayer(x+1,y+1)) * 0.25f;

    } else if (cy == 0 && cx == 1) {
        // ── Green pixel in red row ─────────────────────────────────────────
        g = bayer(x, y);
        r = (bayer(x-1,y) + bayer(x+1,y)) * 0.5f;
        b = (bayer(x,y-1) + bayer(x,y+1)) * 0.5f;

    } else if (cy == 1 && cx == 0) {
        // ── Green pixel in blue row ────────────────────────────────────────
        g = bayer(x, y);
        b = (bayer(x-1,y) + bayer(x+1,y)) * 0.5f;
        r = (bayer(x,y-1) + bayer(x,y+1)) * 0.5f;

    } else {
        // ── Blue pixel ─────────────────────────────────────────────────────
        b = bayer(x, y);
        g = (bayer(x-1,y) + bayer(x+1,y) + bayer(x,y-1) + bayer(x,y+1)) * 0.25f;
        r = (bayer(x-1,y-1) + bayer(x+1,y-1) + bayer(x-1,y+1) + bayer(x+1,y+1)) * 0.25f;
    }

    rgbaTex.write(float4(r, g, b, 1.0f), gid);
}

// ── Accumulate a translated frame into the running sum ────────────────────
//
// 'offset' is the (dx, dy) shift (in pixels) computed by the registrar.
// We sample the input with bilinear filtering at the shifted coordinate.

kernel void accumulateKernel(
    texture2d<float, access::sample>     inputTex [[texture(0)]],
    texture2d<float, access::read_write> accumTex [[texture(1)]],
    constant float2&                     offset   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = accumTex.get_width();
    const uint H = accumTex.get_height();
    if (gid.x >= W || gid.y >= H) return;

    // Map accumulator pixel back to input pixel space
    float2 sampleCoord = float2(gid) - offset + 0.5f;
    float2 normCoord   = sampleCoord / float2(W, H);

    constexpr sampler s(coord::normalized,
                        filter::linear,
                        address::clamp_to_edge);
    float4 inputVal = inputTex.sample(s, normCoord);
    float4 accVal   = accumTex.read(gid);
    accumTex.write(accVal + inputVal, gid);
}

// ── Normalize accumulator by dividing by frameCount ────────────────────────

kernel void normalizeKernel(
    texture2d<float, access::read_write> accumTex   [[texture(0)]],
    constant float&                      frameCount [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= accumTex.get_width() || gid.y >= accumTex.get_height()) return;
    float4 val = accumTex.read(gid);
    accumTex.write(val / max(frameCount, 1.0f), gid);
}
"""

// MARK: - Bayer Pattern → Red-pixel offset

private func redOffset(for pattern: String) -> (Int32, Int32) {
    switch pattern.uppercased() {
    case "RGGB": return (0, 0)
    case "BGGR": return (1, 1)
    case "GRBG": return (1, 0)
    case "GBRG": return (0, 1)
    default:     return (0, 0)   // default to RGGB
    }
}

// MARK: - MetalProcessor

final class MetalProcessor {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let debayerPSO:    MTLComputePipelineState
    private let accumulatePSO: MTLComputePipelineState
    private let normalizePSO:  MTLComputePipelineState

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw PipelineError.metalNotAvailable
        }
        guard let queue = dev.makeCommandQueue() else {
            throw PipelineError.metalNotAvailable
        }
        device = dev
        commandQueue = queue

        // Compile embedded shaders once at startup
        let library = try dev.makeLibrary(source: kShaderSource, options: nil)

        func makePSO(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw PipelineError.processingError("Metal function '\(name)' not found.")
            }
            return try dev.makeComputePipelineState(function: fn)
        }

        debayerPSO    = try makePSO("debayerKernel")
        accumulatePSO = try makePSO("accumulateKernel")
        normalizePSO  = try makePSO("normalizeKernel")
    }

    // MARK: - Debayer

    /// Convert a single-channel raw Bayer image to RGBA float32.
    /// Returns a flat array of Float in RGBA interleaved order, row-major.
    func debayer(rawData: [Float],
                 width: Int,
                 height: Int,
                 pattern: String) throws -> [Float] {

        // Upload raw (Bayer) data to a single-channel R32Float texture
        let bayerDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        bayerDesc.usage = .shaderRead
        bayerDesc.storageMode = .shared   // unified memory – no copy needed on Apple Silicon

        guard let bayerTex = device.makeTexture(descriptor: bayerDesc) else {
            throw PipelineError.processingError("Could not create Bayer texture.")
        }
        rawData.withUnsafeBytes { ptr in
            bayerTex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: width * MemoryLayout<Float>.size)
        }

        // Output RGBA32Float texture
        let rgbaDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        rgbaDesc.usage = [.shaderRead, .shaderWrite]
        rgbaDesc.storageMode = .shared

        guard let rgbaTex = device.makeTexture(descriptor: rgbaDesc) else {
            throw PipelineError.processingError("Could not create RGBA texture.")
        }

        // Encode and run the debayer kernel
        guard let cmdBuf  = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw PipelineError.processingError("Could not create Metal command buffer.")
        }

        encoder.setComputePipelineState(debayerPSO)
        encoder.setTexture(bayerTex, index: 0)
        encoder.setTexture(rgbaTex,  index: 1)

        var (rx, ry) = redOffset(for: pattern)
        var offsetVec = SIMD2<Int32>(rx, ry)
        encoder.setBytes(&offsetVec, length: MemoryLayout<SIMD2<Int32>>.size, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width:  (width  + 15) / 16,
            height: (height + 15) / 16,
            depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back the RGBA texture from shared memory (no extra copy on Apple Silicon)
        var result = [Float](repeating: 0, count: width * height * 4)
        result.withUnsafeMutableBytes { ptr in
            rgbaTex.getBytes(
                ptr.baseAddress!,
                bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                from: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0)
        }
        return result
    }

    // MARK: - Stack

    /// Stack an array of RGBA float32 images (each `width × height × 4` floats)
    /// using the provided per-frame translations.  Returns the mean image.
    func stack(images: [[Float]],
               transforms: [FrameTransform],
               width: Int,
               height: Int) throws -> [Float] {

        guard !images.isEmpty else {
            throw PipelineError.processingError("No images to stack.")
        }

        // Create the float32 accumulator texture (read_write)
        let accumDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        accumDesc.usage = [.shaderRead, .shaderWrite]
        accumDesc.storageMode = .shared

        guard let accumTex = device.makeTexture(descriptor: accumDesc) else {
            throw PipelineError.processingError("Could not create accumulator texture.")
        }

        // Zero-initialise by writing zeroes
        let zeroData = [Float](repeating: 0, count: width * height * 4)
        zeroData.withUnsafeBytes { ptr in
            accumTex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: width * 4 * MemoryLayout<Float>.size)
        }

        let inputDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        inputDesc.usage = [.shaderRead, .shaderSample]
        inputDesc.storageMode = .shared

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: (width+15)/16, height: (height+15)/16, depth: 1)

        for (idx, imageData) in images.enumerated() {
            guard let inputTex = device.makeTexture(descriptor: inputDesc) else { continue }
            imageData.withUnsafeBytes { ptr in
                inputTex.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: ptr.baseAddress!,
                    bytesPerRow: width * 4 * MemoryLayout<Float>.size)
            }

            guard let cmdBuf  = commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(accumulatePSO)
            encoder.setTexture(inputTex, index: 0)
            encoder.setTexture(accumTex, index: 1)

            let t = idx < transforms.count ? transforms[idx] : .identity
            var shift = SIMD2<Float>(t.dx, t.dy)
            encoder.setBytes(&shift, length: MemoryLayout<SIMD2<Float>>.size, index: 0)

            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }

        // Normalize
        guard let normBuf  = commandQueue.makeCommandBuffer(),
              let normEnc  = normBuf.makeComputeCommandEncoder() else {
            throw PipelineError.processingError("Could not create normalize command.")
        }
        normEnc.setComputePipelineState(normalizePSO)
        normEnc.setTexture(accumTex, index: 0)
        var frameCount = Float(images.count)
        normEnc.setBytes(&frameCount, length: MemoryLayout<Float>.size, index: 0)
        normEnc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
        normEnc.endEncoding()
        normBuf.commit()
        normBuf.waitUntilCompleted()

        // Read back result
        var result = [Float](repeating: 0, count: width * height * 4)
        result.withUnsafeMutableBytes { ptr in
            accumTex.getBytes(
                ptr.baseAddress!,
                bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                from: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0)
        }
        return result
    }
}
