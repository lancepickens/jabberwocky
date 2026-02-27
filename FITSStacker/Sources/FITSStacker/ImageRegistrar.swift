// ImageRegistrar.swift – Compute per-frame translation transforms
//
// Strategy (translation-only; sufficient for tracked mounts + Seestar S50):
//
//   1. Convert each debayered RGBA image to a luminance plane.
//   2. Detect stellar centroids in the reference frame and each subsequent frame.
//   3. Match centroids by proximity (nearest-neighbour) with a spatial guard.
//   4. Compute the *median* (dx, dy) over all matched pairs for robustness
//      against mis-matches and variable star brightness.
//
// Runs on the CPU using Accelerate for the luminance conversion (vDSP).
// The registration itself is fast enough to run in parallel per-frame using
// Swift Concurrency (async/await + withTaskGroup).

import Foundation
import Accelerate

final class ImageRegistrar {

    private let starDetector = StarDetector()

    // MARK: - Public API

    /// Register an array of debayered RGBA images to the first (reference) frame.
    ///
    /// - Parameters:
    ///   - images:   Each element is a row-major RGBA Float32 array of `width × height × 4`.
    ///   - width:    Image width in pixels.
    ///   - height:   Image height in pixels.
    ///   - progress: Called with values in [0, 1] as frames are processed.
    /// - Returns: One `FrameTransform` per input image (index 0 is always `.identity`).
    func register(images: [[Float]],
                  width: Int,
                  height: Int,
                  progress: ((Double) -> Void)? = nil) async throws -> [FrameTransform] {

        guard !images.isEmpty else { return [] }

        // Convert reference to luminance and detect its stars
        let refLum   = luminance(from: images[0], width: width, height: height)
        let refStars = starDetector.detect(in: refLum, width: width, height: height)

        var transforms = [FrameTransform](repeating: .identity, count: images.count)

        // Process frames concurrently (Apple Silicon has many performance cores)
        await withTaskGroup(of: (Int, FrameTransform).self) { group in
            for idx in 1 ..< images.count {
                let image = images[idx]
                group.addTask {
                    let lum   = self.luminance(from: image, width: width, height: height)
                    let stars = self.starDetector.detect(in: lum, width: width, height: height)
                    let xf    = self.computeTransform(refStars: refStars, frameStars: stars)
                    return (idx, xf)
                }
            }
            var completed = 0
            for await (idx, xf) in group {
                transforms[idx] = xf
                completed += 1
                progress?(Double(completed) / Double(images.count - 1))
            }
        }

        return transforms
    }

    // MARK: - Private helpers

    /// Convert a width×height RGBA float32 image to luminance using ITU-R BT.709.
    private func luminance(from rgba: [Float], width: Int, height: Int) -> [Float] {
        let pixCount = width * height
        var lum = [Float](repeating: 0, count: pixCount)

        // Extract R, G, B planes then form L = 0.2126R + 0.7152G + 0.0722B
        // rgba layout: R G B A  R G B A  ...  (stride 4)
        var rPlane = [Float](repeating: 0, count: pixCount)
        var gPlane = [Float](repeating: 0, count: pixCount)
        var bPlane = [Float](repeating: 0, count: pixCount)

        rgba.withUnsafeBufferPointer { buf in
            cblas_scopy(Int32(pixCount), buf.baseAddress! + 0, 4, &rPlane, 1)
            cblas_scopy(Int32(pixCount), buf.baseAddress! + 1, 4, &gPlane, 1)
            cblas_scopy(Int32(pixCount), buf.baseAddress! + 2, 4, &bPlane, 1)
        }

        var rWeight: Float = 0.2126, gWeight: Float = 0.7152, bWeight: Float = 0.0722

        // lum = rPlane * 0.2126
        vDSP_vsmul(rPlane, 1, &rWeight, &lum, 1, vDSP_Length(pixCount))
        // lum += gPlane * 0.7152
        vDSP_vsma(gPlane, 1, &gWeight, lum, 1, &lum, 1, vDSP_Length(pixCount))
        // lum += bPlane * 0.0722
        vDSP_vsma(bPlane, 1, &bWeight, lum, 1, &lum, 1, vDSP_Length(pixCount))

        return lum
    }

    /// Match stars and return the median translation.
    private func computeTransform(refStars: [Star], frameStars: [Star]) -> FrameTransform {
        guard !refStars.isEmpty && !frameStars.isEmpty else { return .identity }

        // Build reference k-d structure (simple grid bucket for speed)
        // For typical star counts (< 60) an O(n²) scan is perfectly fast.
        let searchRadius: Float = 80.0

        var dxList: [Float] = []
        var dyList: [Float] = []
        dxList.reserveCapacity(frameStars.count)
        dyList.reserveCapacity(frameStars.count)

        for fStar in frameStars {
            var bestDist = searchRadius * searchRadius
            var bestDx: Float = 0
            var bestDy: Float = 0
            var found = false

            for rStar in refStars {
                let ddx = fStar.x - rStar.x
                let ddy = fStar.y - rStar.y
                let dist2 = ddx * ddx + ddy * ddy
                if dist2 < bestDist {
                    bestDist = dist2
                    // Translation to apply to fStar to reach rStar position
                    bestDx = rStar.x - fStar.x
                    bestDy = rStar.y - fStar.y
                    found = true
                }
            }
            if found {
                dxList.append(bestDx)
                dyList.append(bestDy)
            }
        }

        guard !dxList.isEmpty else { return .identity }

        // Median is robust against outliers
        let dx = median(of: dxList)
        let dy = median(of: dyList)
        return FrameTransform(dx: dx, dy: dy)
    }

    private func median(of values: [Float]) -> Float {
        var sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[mid - 1] + sorted[mid]) / 2
        }
        return sorted[mid]
    }
}
