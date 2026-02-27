// FITSWriter.swift – Write stacked images to disk
//
// Outputs:
//   • 16-bit linear TIFF  – universal import into PixInsight, Siril, APP, etc.
//   • FITS (BITPIX=-32)   – 32-bit float FITS for downstream processing
//
// Both formats preserve the full linear dynamic range without gamma encoding.
// The RGBA Metal texture data is converted to 3-channel RGB for output.

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Accelerate

enum FITSWriterError: LocalizedError {
    case cgImageCreationFailed
    case tiffWriteFailed
    case fitWriteFailed(String)

    var errorDescription: String? {
        switch self {
        case .cgImageCreationFailed: return "Could not create CGImage for TIFF output."
        case .tiffWriteFailed:       return "Failed to write TIFF file."
        case .fitWriteFailed(let m): return "Failed to write FITS file: \(m)"
        }
    }
}

final class FITSWriter {

    // MARK: - 16-bit TIFF

    /// Save an RGBA float32 image as a 16-bit per channel linear RGB TIFF.
    /// - Parameters:
    ///   - rgba:   Row-major RGBA float32, `width × height × 4`.
    ///   - width:  Image width in pixels.
    ///   - height: Image height in pixels.
    ///   - to:     Destination file URL.
    static func saveTIFF(rgba: [Float],
                         width: Int,
                         height: Int,
                         to url: URL) throws {

        let pixCount = width * height

        // Convert float [0,1] → UInt16, strip alpha, produce RGB16 row-major buffer
        var rgb16 = [UInt16](repeating: 0, count: pixCount * 3)

        for i in 0..<pixCount {
            let r = rgba[i * 4 + 0]
            let g = rgba[i * 4 + 1]
            let b = rgba[i * 4 + 2]
            rgb16[i * 3 + 0] = UInt16(clamp(r) * 65535)
            rgb16[i * 3 + 1] = UInt16(clamp(g) * 65535)
            rgb16[i * 3 + 2] = UInt16(clamp(b) * 65535)
        }

        let bytesPerRow = width * 3 * MemoryLayout<UInt16>.size
        let dataSize    = height * bytesPerRow
        let cfData      = Data(bytes: rgb16, count: dataSize) as CFData

        guard let provider = CGDataProvider(data: cfData),
              let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB) else {
            throw FITSWriterError.cgImageCreationFailed
        }

        // CGImage: 16-bit per component, 3 components, no alpha
        let bitmapInfo = CGBitmapInfo(rawValue:
            CGBitmapInfo.byteOrder16Little.rawValue |
            CGImageAlphaInfo.none.rawValue)

        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 16,
            bitsPerPixel: 48,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent) else {
            throw FITSWriterError.cgImageCreationFailed
        }

        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.tiff.identifier as CFString,
            1, nil) else {
            throw FITSWriterError.tiffWriteFailed
        }

        // Embed DPI and linear colour profile metadata
        let props: CFDictionary = [
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFXResolution: 72,
                kCGImagePropertyTIFFYResolution: 72
            ],
            kCGImagePropertyColorModel: kCGImagePropertyColorModelRGB
        ] as CFDictionary

        CGImageDestinationAddImage(dest, cgImage, props)
        guard CGImageDestinationFinalize(dest) else {
            throw FITSWriterError.tiffWriteFailed
        }
    }

    // MARK: - FITS (BITPIX=-32, 3-plane)

    /// Write the stacked RGB data as a simple FITS file with three image planes
    /// (NAXIS=3, NAXIS3=3).  Uses BITPIX=-32 (IEEE float, big-endian).
    static func saveFITS(rgba: [Float],
                         width: Int,
                         height: Int,
                         frameCount: Int,
                         to url: URL) throws {

        var data = Data()

        // ── Header ────────────────────────────────────────────────────────────
        let headerCards: [String] = [
            fitsCard("SIMPLE",  value: "T",          comment: "FITS standard"),
            fitsCard("BITPIX",  value: "-32",         comment: "IEEE 32-bit float"),
            fitsCard("NAXIS",   value: "3",           comment: "3 image planes"),
            fitsCard("NAXIS1",  value: "\(width)",    comment: "Width"),
            fitsCard("NAXIS2",  value: "\(height)",   comment: "Height"),
            fitsCard("NAXIS3",  value: "3",           comment: "R G B planes"),
            fitsCard("BSCALE",  value: "1.0",         comment: ""),
            fitsCard("BZERO",   value: "0.0",         comment: ""),
            fitsCard("STACKN",  value: "\(frameCount)", comment: "Frames stacked"),
            fitsCard("CREATOR", value: "'FITSStacker'", comment: "Apple Silicon macOS app"),
            fitsCard("DATE",    value: "'\(isoDate())'", comment: "Stack creation date"),
            "END" + String(repeating: " ", count: 77)
        ]

        var headerBytes = [UInt8]()
        for card in headerCards {
            var padded = card
            if padded.count < 80 { padded += String(repeating: " ", count: 80 - padded.count) }
            headerBytes.append(contentsOf: padded.utf8.prefix(80))
        }

        // Pad header to next 2880-byte boundary
        let headerBlocks = (headerBytes.count + 2879) / 2880
        headerBytes += [UInt8](repeating: UInt8(ascii: " "), count: headerBlocks * 2880 - headerBytes.count)
        data.append(contentsOf: headerBytes)

        // ── Data (big-endian float32, R plane, G plane, B plane) ─────────────
        let pixCount = width * height
        var planeData = [UInt8]()
        planeData.reserveCapacity(pixCount * 3 * 4)

        for plane in 0..<3 {
            for i in 0..<pixCount {
                let val  = rgba[i * 4 + plane]
                let bits = val.bitPattern.bigEndian
                planeData.append(UInt8((bits >> 24) & 0xFF))
                planeData.append(UInt8((bits >> 16) & 0xFF))
                planeData.append(UInt8((bits >>  8) & 0xFF))
                planeData.append(UInt8( bits        & 0xFF))
            }
        }

        // Pad data to 2880-byte boundary
        let dataBlocks = (planeData.count + 2879) / 2880
        planeData += [UInt8](repeating: 0, count: dataBlocks * 2880 - planeData.count)
        data.append(contentsOf: planeData)

        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw FITSWriterError.fitWriteFailed(error.localizedDescription)
        }
    }

    // MARK: - Private helpers

    private static func clamp(_ v: Float) -> Float { Swift.max(0, Swift.min(1, v)) }

    private static func fitsCard(_ keyword: String, value: String, comment: String) -> String {
        let kw    = keyword.padding(toLength: 8, withPad: " ", startingAt: 0)
        let vc    = "= \(value)"
        let full  = "\(kw)\(vc)"
        let avail = 80 - full.count - 3  // leave room for " / comment"
        if comment.isEmpty || avail <= 0 {
            return full.padding(toLength: 80, withPad: " ", startingAt: 0)
        }
        let trimmed = String(comment.prefix(avail))
        return (full + " / " + trimmed).padding(toLength: 80, withPad: " ", startingAt: 0)
    }

    private static func isoDate() -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        return df.string(from: Date())
    }
}
