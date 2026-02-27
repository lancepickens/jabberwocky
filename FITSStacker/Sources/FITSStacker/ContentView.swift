// ContentView.swift – Main application window (SwiftUI, macOS 14+)
//
// Layout:
//   ┌─────────────────────────────────────────────┐
//   │  FITSStacker    [Seestar S50 FITS Stacker]  │
//   │─────────────────────────────────────────────│
//   │  Home Directory: [path…]  [Choose…]         │
//   │  lights/    process/    output/   (badges)   │
//   │─────────────────────────────────────────────│
//   │  [▶ Stack Frames]          [Stop]            │
//   │  ████████░░░░░░░░  42 %                      │
//   │  Status: Debayering 7 / 32                   │
//   │─────────────────────────────────────────────│
//   │  Log ▼                                       │
//   │  [12:34:01] Found 32 FITS files              │
//   │  …                                           │
//   │─────────────────────────────────────────────│
//   │  Output: stacked_20250414_123456.tiff  [📂]  │
//   └─────────────────────────────────────────────┘

import SwiftUI

struct ContentView: View {

    @StateObject private var pipeline = StackingPipeline()
    @State private var showDirectoryChooser = false
    @State private var logScrollProxy: ScrollViewProxy? = nil

    // Directory structure status
    private var lightsOK: Bool {
        guard let home = pipeline.homeDirectory else { return false }
        return FileManager.default.fileExists(
            atPath: home.appendingPathComponent("lights").path)
    }
    private var processExists: Bool {
        guard let home = pipeline.homeDirectory else { return false }
        return FileManager.default.fileExists(
            atPath: home.appendingPathComponent("process").path)
    }
    private var outputExists: Bool {
        guard let home = pipeline.homeDirectory else { return false }
        return FileManager.default.fileExists(
            atPath: home.appendingPathComponent("output").path)
    }

    // MARK: - Body

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            headerBar
            Divider()
            directorySection
            Divider()
            controlSection
            Divider()
            logSection
            if pipeline.outputURL != nil {
                Divider()
                outputSection
            }
        }
        .frame(minWidth: 640, minHeight: 540)
        .background(Color(NSColor.windowBackgroundColor))
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack {
            Image(systemName: "sparkles")
                .font(.title)
                .foregroundColor(.accentColor)
            VStack(alignment: .leading, spacing: 1) {
                Text("FITSStacker")
                    .font(.headline)
                Text("Seestar S50 · Apple Silicon · GPU-accelerated")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    // MARK: - Directory Section

    private var directorySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Home Directory")
                .font(.subheadline)
                .foregroundColor(.secondary)

            HStack {
                // Path field (tappable)
                Text(pipeline.homeDirectory?.path ?? "No directory selected")
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(pipeline.homeDirectory == nil ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(6)
                    .background(Color(NSColor.textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))

                Button("Choose…") {
                    chooseDirectory()
                }
                .disabled(pipeline.isRunning)
            }

            // Folder badges
            HStack(spacing: 12) {
                folderBadge("lights",  ok: lightsOK,      required: true)
                folderBadge("process", ok: processExists,  required: false)
                folderBadge("output",  ok: outputExists,   required: false)
                Spacer()
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    private func folderBadge(_ name: String, ok: Bool, required: Bool) -> some View {
        HStack(spacing: 4) {
            Image(systemName: ok ? "folder.fill" : (required ? "folder.badge.questionmark" : "folder"))
                .foregroundColor(ok ? .green : (required ? .orange : .secondary))
            Text(name + "/")
                .font(.caption)
                .foregroundColor(ok ? .primary : .secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Control Section

    private var controlSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Action buttons
            HStack(spacing: 12) {
                Button {
                    pipeline.run()
                } label: {
                    Label("Stack Frames", systemImage: "play.fill")
                        .frame(minWidth: 130)
                }
                .buttonStyle(.borderedProminent)
                .disabled(pipeline.isRunning || !lightsOK)
                .keyboardShortcut(.return, modifiers: .command)

                if pipeline.isRunning {
                    Button {
                        pipeline.cancel()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }

                Spacer()
            }

            // Progress bar
            if pipeline.isRunning || pipeline.progress > 0 {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: pipeline.progress)
                        .progressViewStyle(.linear)
                        .tint(pipeline.progress >= 1.0 ? .green : .accentColor)

                    Text(pipeline.statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
            } else {
                Text(pipeline.statusMessage)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Error banner
            if let err = pipeline.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                    Text(err)
                        .font(.caption)
                        .foregroundColor(.red)
                }
                .padding(8)
                .background(Color.red.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    // MARK: - Log Section

    private var logSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Log")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                if !pipeline.logEntries.isEmpty {
                    Button("Clear") { pipeline.logEntries.removeAll() }
                        .font(.caption)
                        .buttonStyle(.plain)
                        .foregroundColor(.accentColor)
                }
            }

            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(pipeline.logEntries) { entry in
                            Text(entry.formatted)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.primary.opacity(0.85))
                                .textSelection(.enabled)
                                .id(entry.id)
                        }
                    }
                    .padding(6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(minHeight: 140, maxHeight: 220)
                .background(Color(NSColor.textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .onAppear { logScrollProxy = proxy }
                .onChange(of: pipeline.logEntries.count) { _ in
                    if let last = pipeline.logEntries.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    // MARK: - Output Section

    private var outputSection: some View {
        HStack {
            Image(systemName: "checkmark.seal.fill")
                .foregroundColor(.green)
            if let out = pipeline.outputURL {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Output saved")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(out.lastPathComponent)
                        .font(.system(.body, design: .monospaced))
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                Button {
                    NSWorkspace.shared.selectFile(out.path,
                        inFileViewerRootedAtPath: out.deletingLastPathComponent().path)
                } label: {
                    Label("Show in Finder", systemImage: "folder")
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color.green.opacity(0.08))
    }

    // MARK: - Directory Chooser

    private func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.title            = "Select Home Directory"
        panel.message          = "Choose the folder containing lights/, process/, and output/ sub-folders."
        panel.canChooseFiles   = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt           = "Select"

        if panel.runModal() == .OK, let url = panel.url {
            pipeline.homeDirectory = url
            pipeline.progress      = 0
            pipeline.outputURL     = nil
            pipeline.errorMessage  = nil
            pipeline.statusMessage = "Ready – press ⌘↵ to start stacking."

            // Auto-create expected sub-folders so the badges show correctly
            let fm = FileManager.default
            for sub in ["lights", "process", "output"] {
                let dir = url.appendingPathComponent(sub)
                if !fm.fileExists(atPath: dir.path) {
                    // Don't create lights/ – user must supply light frames
                    if sub != "lights" {
                        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
                    }
                }
            }
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
        .frame(width: 640, height: 600)
}
