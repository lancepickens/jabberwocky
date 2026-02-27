// Models.swift – Core data types for FITSStacker
// Targets Apple Silicon (arm64) macOS 14+

import Foundation

// MARK: - FITS Header

struct FITSHeader {
    var bitpix: Int = 16
    var width: Int = 0
    var height: Int = 0
    var bayerPattern: String = "RGGB"
    var expTime: Double = 0.0
    var gain: Double = 0.0
    var temperature: Double = 0.0
    /// BZERO offset applied when converting integer pixel values to physical values.
    /// Seestar S50 uses BZERO = 32768 so that INT16 maps to UINT16 range.
    var bzero: Double = 32768.0
    /// BSCALE factor (almost always 1.0)
    var bscale: Double = 1.0
    var keywords: [String: String] = [:]
}

// MARK: - FITS Image

/// Raw FITS image loaded from disk, pixel data normalized to [0, 1].
final class FITSImage {
    let header: FITSHeader
    /// Single-channel Bayer raw pixel data in row-major order, normalized to [0, 1].
    let rawData: [Float]
    let url: URL

    var width: Int { header.width }
    var height: Int { header.height }

    init(header: FITSHeader, rawData: [Float], url: URL) {
        self.header = header
        self.rawData = rawData
        self.url = url
    }
}

// MARK: - Star (for registration)

struct Star {
    let x: Float
    let y: Float
    let brightness: Float
}

// MARK: - Frame Transform (translation only)

struct FrameTransform {
    /// Pixel shift in X to align this frame to the reference.
    let dx: Float
    /// Pixel shift in Y to align this frame to the reference.
    let dy: Float

    static let identity = FrameTransform(dx: 0, dy: 0)
}

// MARK: - Pipeline Errors

enum PipelineError: LocalizedError {
    case missingFolder(String)
    case noFITSFiles
    case metalNotAvailable
    case processingError(String)

    var errorDescription: String? {
        switch self {
        case .missingFolder(let name):
            return "Missing '\(name)' folder in the home directory."
        case .noFITSFiles:
            return "No FITS files (.fits / .fit / .fts) found in the lights folder."
        case .metalNotAvailable:
            return "Metal GPU is not available on this system."
        case .processingError(let msg):
            return msg
        }
    }
}

// MARK: - Log Entry

struct LogEntry: Identifiable {
    let id = UUID()
    let timestamp: Date
    let message: String

    init(_ message: String) {
        self.timestamp = Date()
        self.message = message
    }

    var formatted: String {
        let tf = DateFormatter()
        tf.dateFormat = "HH:mm:ss"
        return "[\(tf.string(from: timestamp))] \(message)"
    }
}
