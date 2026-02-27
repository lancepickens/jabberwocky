// StackingPipeline.swift – Orchestrates the full lights→stacked-output workflow
//
// Observed on the UI thread via @Published; heavy work is dispatched with
// Swift Concurrency (async/await) so the UI stays responsive.
//
// Workflow:
//   1. Verify directory structure (lights / process / output)
//   2. Find FITS files in lights/
//   3. Load each FITS file (FITSReader)
//   4. Debayer each frame on GPU (MetalProcessor)
//   5. Register frames to reference via star matching (ImageRegistrar)
//   6. Stack registered frames on GPU (MetalProcessor)
//   7. Write 16-bit TIFF + FITS to output/

import Foundation
import Combine

@MainActor
final class StackingPipeline: ObservableObject {

    // ── Published state ──────────────────────────────────────────────────────
    @Published var homeDirectory: URL? = nil
    @Published var isRunning: Bool = false
    @Published var progress: Double = 0          // 0…1
    @Published var statusMessage: String = "Ready – select a home directory to begin."
    @Published var logEntries: [LogEntry] = []
    @Published var outputURL: URL? = nil
    @Published var errorMessage: String? = nil

    // MARK: - Run

    func run() {
        guard !isRunning else { return }
        guard let homeDir = homeDirectory else {
            errorMessage = "Please select a home directory first."
            return
        }

        isRunning  = true
        progress   = 0
        logEntries = []
        outputURL  = nil
        errorMessage = nil

        Task {
            do {
                try await runPipeline(homeDir: homeDir)
            } catch {
                log("ERROR: \(error.localizedDescription)")
                errorMessage = error.localizedDescription
                status("Failed – see log for details.")
            }
            isRunning = false
        }
    }

    func cancel() {
        // Swift Task cancellation; processors check Task.isCancelled
        isRunning = false
        status("Cancelled.")
    }

    // MARK: - Pipeline

    private func runPipeline(homeDir: URL) async throws {
        let fm = FileManager.default

        // ── Step 1: Verify / create folders ─────────────────────────────────
        log("Checking directory structure…")
        let lightsDir  = homeDir.appendingPathComponent("lights")
        let processDir = homeDir.appendingPathComponent("process")
        let outputDir  = homeDir.appendingPathComponent("output")

        guard fm.fileExists(atPath: lightsDir.path) else {
            throw PipelineError.missingFolder("lights")
        }
        try fm.createDirectory(at: processDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: outputDir,  withIntermediateDirectories: true)
        log("Folders OK – lights: \(lightsDir.path)")

        // ── Step 2: Find FITS files ───────────────────────────────────────────
        status("Scanning lights folder…")
        let fitsURLs = try findFITSFiles(in: lightsDir)
        guard !fitsURLs.isEmpty else { throw PipelineError.noFITSFiles }
        log("Found \(fitsURLs.count) FITS file(s)")
        setProgress(0.02)

        // ── Step 3: Initialise Metal ──────────────────────────────────────────
        status("Initialising Metal GPU…")
        log("Setting up Metal compute pipeline…")
        let metal = try MetalProcessor()
        log("Metal GPU ready")

        // ── Step 4: Load + Debayer all frames ─────────────────────────────────
        var debayeredFrames: [[Float]] = []
        var imageWidth  = 0
        var imageHeight = 0
        let bayerPat: String

        log("Loading and debayering \(fitsURLs.count) frame(s) on GPU…")
        for (i, url) in fitsURLs.enumerated() {
            try Task.checkCancellation()
            status("Debayering \(i+1) / \(fitsURLs.count):  \(url.lastPathComponent)")

            let fits = try FITSReader.read(from: url)

            if i == 0 {
                imageWidth  = fits.width
                imageHeight = fits.height
                bayerPat    = fits.header.bayerPattern
                log("Image size: \(imageWidth) × \(imageHeight), Bayer: \(bayerPat)")
            }

            // Sanity-check: all frames must match the reference size
            guard fits.width == imageWidth && fits.height == imageHeight else {
                log("WARNING: \(url.lastPathComponent) size mismatch – skipping.")
                continue
            }

            let rgba = try metal.debayer(
                rawData: fits.rawData,
                width:   fits.width,
                height:  fits.height,
                pattern: fits.header.bayerPattern)
            debayeredFrames.append(rgba)

            let p = 0.02 + 0.38 * Double(i + 1) / Double(fitsURLs.count)
            setProgress(p)
        }

        guard !debayeredFrames.isEmpty else {
            throw PipelineError.processingError("No frames loaded successfully.")
        }
        log("Debayered \(debayeredFrames.count) frame(s)")

        // ── Step 5: Save debayered frames to process/ for inspection ──────────
        // (Saves only if ≤ 5 frames to avoid flooding disk)
        if debayeredFrames.count <= 5 {
            for (i, rgba) in debayeredFrames.enumerated() {
                let dstURL = processDir.appendingPathComponent("debayered_\(String(format: "%04d", i)).tiff")
                try? FITSWriter.saveTIFF(rgba: rgba, width: imageWidth, height: imageHeight, to: dstURL)
            }
            log("Saved debayered previews to process/")
        }
        setProgress(0.42)

        // ── Step 6: Register frames ───────────────────────────────────────────
        status("Registering frames…")
        log("Detecting stars and computing transforms…")
        let registrar  = ImageRegistrar()
        let transforms = try await registrar.register(
            images:  debayeredFrames,
            width:   imageWidth,
            height:  imageHeight) { [weak self] prog in
                Task { @MainActor [weak self] in
                    self?.setProgress(0.42 + 0.28 * prog)
                    self?.status("Registering: \(Int(prog * 100))%")
                }
            }

        let nonZero = transforms.filter { $0.dx != 0 || $0.dy != 0 }.count
        log("Registration complete – \(nonZero) frame(s) shifted")
        setProgress(0.70)

        // ── Step 7: Stack ─────────────────────────────────────────────────────
        status("Stacking \(debayeredFrames.count) frames on GPU…")
        log("Stacking with per-frame translation compensation…")
        let stacked = try metal.stack(
            images:     debayeredFrames,
            transforms: transforms,
            width:      imageWidth,
            height:     imageHeight)
        log("Stack complete")
        setProgress(0.90)

        // ── Step 8: Write output ──────────────────────────────────────────────
        status("Writing output files…")
        let ts = formattedTimestamp()
        let tiffURL = outputDir.appendingPathComponent("stacked_\(ts).tiff")
        let fitsOut = outputDir.appendingPathComponent("stacked_\(ts).fits")

        try FITSWriter.saveTIFF(rgba: stacked, width: imageWidth, height: imageHeight, to: tiffURL)
        log("Saved TIFF:  \(tiffURL.lastPathComponent)")

        try FITSWriter.saveFITS(rgba: stacked, width: imageWidth, height: imageHeight,
                                frameCount: debayeredFrames.count, to: fitsOut)
        log("Saved FITS:  \(fitsOut.lastPathComponent)")

        setProgress(1.0)
        outputURL = tiffURL
        status("Done! \(debayeredFrames.count) frames stacked → \(tiffURL.lastPathComponent)")
        log("━━━ Pipeline complete ━━━")
    }

    // MARK: - Helpers

    private func findFITSFiles(in directory: URL) throws -> [URL] {
        let exts = Set(["fits", "fit", "fts"])
        return try FileManager.default
            .contentsOfDirectory(at: directory,
                                 includingPropertiesForKeys: [.fileSizeKey],
                                 options: .skipsHiddenFiles)
            .filter { exts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func log(_ message: String) {
        logEntries.append(LogEntry(message))
    }

    private func status(_ message: String) {
        statusMessage = message
    }

    private func setProgress(_ value: Double) {
        progress = min(max(value, 0), 1)
    }

    private func formattedTimestamp() -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd_HHmmss"
        return df.string(from: Date())
    }
}
