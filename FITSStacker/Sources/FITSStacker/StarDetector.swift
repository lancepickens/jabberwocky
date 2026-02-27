// StarDetector.swift – Detect stellar centroids in a luminance image
//
// Algorithm:
//   1. 5×5 Gaussian blur via vImage (reduces hot pixels)
//   2. Adaptive threshold: mean + k×σ  (k ≈ 4)
//   3. Non-maximum suppression in a 7×7 window
//   4. Centroid refinement within a 5×5 window
//   5. Sort by brightness, return top-N stars
//
// All heavy lifting uses Accelerate (vDSP + vImage) for SIMD throughput
// on Apple Silicon.

import Foundation
import Accelerate

final class StarDetector {

    // MARK: - Public API

    /// Detect stars in a normalised [0,1] grayscale image.
    /// - Parameters:
    ///   - image: Row-major Float array, width × height pixels.
    ///   - width: Image width in pixels.
    ///   - height: Image height in pixels.
    ///   - maxStars: Maximum number of stars to return (sorted by brightness).
    ///   - sigmaThreshold: How many σ above the mean to set the detection threshold.
    func detect(in image: [Float],
                width: Int,
                height: Int,
                maxStars: Int = 60,
                sigmaThreshold: Float = 4.0) -> [Star] {

        let blurred = gaussianBlur(image, width: width, height: height)

        // ── Threshold ──────────────────────────────────────────────────────────
        let n = vDSP_Length(width * height)
        var mean: Float = 0
        var stdDev: Float = 0
        vDSP_meanv(blurred, 1, &mean, n)
        // Compute variance = E[x²] - mean²
        var meanSq: Float = 0
        vDSP_measqv(blurred, 1, &meanSq, n)
        stdDev = sqrt(max(meanSq - mean * mean, 0))

        let threshold = mean + sigmaThreshold * stdDev

        // ── Non-maximum suppression (NMS) ─────────────────────────────────────
        let halfWin = 4           // 9×9 NMS window
        var stars: [Star] = []
        stars.reserveCapacity(maxStars * 2)

        for y in halfWin ..< (height - halfWin) {
            for x in halfWin ..< (width - halfWin) {
                let val = blurred[y * width + x]
                guard val >= threshold else { continue }

                // Check local maximum
                var isMax = true
                outer: for dy in -halfWin...halfWin {
                    for dx in -halfWin...halfWin {
                        if dx == 0 && dy == 0 { continue }
                        if blurred[(y + dy) * width + (x + dx)] > val {
                            isMax = false; break outer
                        }
                    }
                }
                guard isMax else { continue }

                // ── Sub-pixel centroid via intensity-weighted average ─────────
                var sumW: Float = 0, sumX: Float = 0, sumY: Float = 0
                for dy in -2...2 {
                    for dx in -2...2 {
                        let nx = x + dx, ny = y + dy
                        guard nx >= 0 && nx < width && ny >= 0 && ny < height else { continue }
                        let w = blurred[ny * width + nx]
                        sumW += w
                        sumX += w * Float(nx)
                        sumY += w * Float(ny)
                    }
                }
                let cx = sumW > 0 ? sumX / sumW : Float(x)
                let cy = sumW > 0 ? sumY / sumW : Float(y)
                stars.append(Star(x: cx, y: cy, brightness: val))
            }
        }

        // Sort descending by brightness and cap
        stars.sort { $0.brightness > $1.brightness }
        return Array(stars.prefix(maxStars))
    }

    // MARK: - Private helpers

    /// 5×5 Gaussian blur using vImage (SIMD-accelerated).
    private func gaussianBlur(_ image: [Float], width: Int, height: Int) -> [Float] {
        // sigma ≈ 1.0 5×5 Gaussian kernel (normalised)
        let kernel: [Float] = [
             1,  4,  7,  4,  1,
             4, 16, 26, 16,  4,
             7, 26, 41, 26,  7,
             4, 16, 26, 16,  4,
             1,  4,  7,  4,  1
        ]
        let kernelSum: Float = kernel.reduce(0, +)
        let normKernel = kernel.map { $0 / kernelSum }

        var output = [Float](repeating: 0, count: width * height)

        image.withUnsafeBufferPointer { srcBuf in
            output.withUnsafeMutableBufferPointer { dstBuf in
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcBuf.baseAddress!),
                    height: vImagePixelCount(height),
                    width:  vImagePixelCount(width),
                    rowBytes: width * MemoryLayout<Float>.size)
                var dst = vImage_Buffer(
                    data: dstBuf.baseAddress!,
                    height: vImagePixelCount(height),
                    width:  vImagePixelCount(width),
                    rowBytes: width * MemoryLayout<Float>.size)
                normKernel.withUnsafeBufferPointer { kBuf in
                    vImageConvolve_PlanarF(
                        &src, &dst, nil, 0, 0,
                        kBuf.baseAddress!, 5, 5,
                        0, vImage_Flags(kvImageEdgeExtend))
                }
            }
        }
        return output
    }
}
