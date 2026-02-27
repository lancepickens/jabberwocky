// FITSReader.swift – Pure-Swift FITS parser
//
// Handles the subset of FITS used by the ZWO Seestar S50:
//   BITPIX = 16 (INT16 big-endian, BZERO=32768, BSCALE=1)
//   NAXIS  = 2 (single Bayer plane)
//   BAYERPAT / BAYER keyword for CFA pattern
//
// Reference: FITS Standard v4.0  https://fits.gsfc.nasa.gov/fits_standard.html
//
// All reads are zero-copy via Data; the hot conversion loop uses SIMD-friendly
// patterns that the compiler can auto-vectorise on Apple Silicon (arm64).

import Foundation
import Accelerate

enum FITSReaderError: LocalizedError {
    case fileTooSmall
    case missingENDCard
    case unsupportedBitpix(Int)
    case insufficientData(need: Int, have: Int)
    case invalidDimensions

    var errorDescription: String? {
        switch self {
        case .fileTooSmall:                         return "File is too small to be a valid FITS file."
        case .missingENDCard:                       return "FITS header END card not found."
        case .unsupportedBitpix(let b):             return "Unsupported BITPIX=\(b). Only 8, 16, 32, -32 are supported."
        case .insufficientData(let n, let h):       return "Expected \(n) bytes of pixel data, only \(h) available."
        case .invalidDimensions:                    return "NAXIS1 or NAXIS2 is zero or missing."
        }
    }
}

final class FITSReader {

    // MARK: - Public API

    static func read(from url: URL) throws -> FITSImage {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try parse(data: data, url: url)
    }

    // MARK: - Parsing

    private static func parse(data: Data, url: URL) throws -> FITSImage {
        guard data.count >= 2880 else { throw FITSReaderError.fileTooSmall }

        var header = FITSHeader()
        var offset = 0
        var endFound = false

        // ── Walk header blocks (each 2880 bytes = 36 × 80-byte cards) ─────────
        while !endFound {
            guard offset + 2880 <= data.count else { throw FITSReaderError.missingENDCard }

            for cardIndex in 0..<36 {
                let base = offset + cardIndex * 80
                // Keyword is the first 8 bytes (ASCII, space-padded)
                let kwBytes = data[base ..< base + 8]
                guard let kw = String(bytes: kwBytes, encoding: .ascii) else { continue }
                let keyword = kw.trimmingCharacters(in: .whitespaces)

                if keyword == "END" { endFound = true; break }

                // Value / comment field starts at byte 10 (after "= ")
                guard data[base + 8] == UInt8(ascii: "=") else { continue }
                let vcBytes = data[base + 10 ..< base + 80]
                guard let vcString = String(bytes: vcBytes, encoding: .ascii) else { continue }
                // Strip inline comment
                let rawValue = (vcString.components(separatedBy: "/").first ?? "")
                    .trimmingCharacters(in: .whitespaces)
                // Strip FITS string quotes
                let value = rawValue.trimmingCharacters(in: CharacterSet(charactersIn: "'")
                    .union(.whitespaces))

                switch keyword {
                case "BITPIX":   header.bitpix  = Int(value) ?? 16
                case "NAXIS1":   header.width   = Int(value) ?? 0
                case "NAXIS2":   header.height  = Int(value) ?? 0
                case "BZERO":    header.bzero   = Double(value) ?? 32768.0
                case "BSCALE":   header.bscale  = Double(value) ?? 1.0
                case "EXPTIME", "EXPOSURE":
                    header.expTime  = Double(value) ?? 0
                case "GAIN":     header.gain     = Double(value) ?? 0
                case "CCD-TEMP": header.temperature = Double(value) ?? 0
                case "BAYERPAT", "BAYER", "COLORTYP":
                    header.bayerPattern = value
                default:
                    header.keywords[keyword] = value
                }
            }

            offset += 2880
        }

        guard header.width > 0 && header.height > 0 else { throw FITSReaderError.invalidDimensions }

        let pixelCount = header.width * header.height
        var floatData = [Float](repeating: 0, count: pixelCount)

        switch header.bitpix {

        // ── 16-bit signed integer (most common for Seestar S50) ───────────────
        case 16:
            let bytesNeeded = pixelCount * 2
            guard offset + bytesNeeded <= data.count else {
                throw FITSReaderError.insufficientData(need: bytesNeeded,
                                                       have: data.count - offset)
            }
            // FITS is big-endian; swap bytes then convert to Float
            var int16Buf = [Int16](repeating: 0, count: pixelCount)
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: offset)
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<pixelCount {
                    int16Buf[i] = Int16(bitPattern: src[i].bigEndian)
                }
            }
            // Apply BSCALE / BZERO: physical = int * bscale + bzero
            let bscale = Float(header.bscale)
            let bzero  = Float(header.bzero)
            vDSP_vsmsa(
                int16Buf.map { Float($0) }, 1,
                [bscale],
                [bzero],
                &floatData, 1,
                vDSP_Length(pixelCount)
            )

        // ── 8-bit unsigned integer ─────────────────────────────────────────────
        case 8:
            let bytesNeeded = pixelCount
            guard offset + bytesNeeded <= data.count else {
                throw FITSReaderError.insufficientData(need: bytesNeeded,
                                                       have: data.count - offset)
            }
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: offset)
                    .assumingMemoryBound(to: UInt8.self)
                for i in 0..<pixelCount {
                    floatData[i] = Float(src[i]) * Float(header.bscale) + Float(header.bzero)
                }
            }

        // ── 32-bit signed integer ─────────────────────────────────────────────
        case 32:
            let bytesNeeded = pixelCount * 4
            guard offset + bytesNeeded <= data.count else {
                throw FITSReaderError.insufficientData(need: bytesNeeded,
                                                       have: data.count - offset)
            }
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: offset)
                    .assumingMemoryBound(to: UInt32.self)
                for i in 0..<pixelCount {
                    floatData[i] = Float(Int32(bitPattern: src[i].bigEndian))
                }
            }

        // ── 32-bit IEEE float ─────────────────────────────────────────────────
        case -32:
            let bytesNeeded = pixelCount * 4
            guard offset + bytesNeeded <= data.count else {
                throw FITSReaderError.insufficientData(need: bytesNeeded,
                                                       have: data.count - offset)
            }
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: offset)
                    .assumingMemoryBound(to: UInt32.self)
                for i in 0..<pixelCount {
                    floatData[i] = Float(bitPattern: src[i].bigEndian)
                }
            }

        default:
            throw FITSReaderError.unsupportedBitpix(header.bitpix)
        }

        // ── Normalize to [0, 1] using vDSP (SIMD-accelerated) ─────────────────
        var minVal: Float = 0
        var maxVal: Float = 0
        vDSP_minv(floatData, 1, &minVal, vDSP_Length(pixelCount))
        vDSP_maxv(floatData, 1, &maxVal, vDSP_Length(pixelCount))

        if maxVal > minVal {
            var offset_neg = -minVal
            let scale = 1.0 / (maxVal - minVal)
            vDSP_vsadd(floatData, 1, &offset_neg, &floatData, 1, vDSP_Length(pixelCount))
            vDSP_vsmul(floatData, 1, [scale], &floatData, 1, vDSP_Length(pixelCount))
        }

        return FITSImage(header: header, rawData: floatData, url: url)
    }
}
