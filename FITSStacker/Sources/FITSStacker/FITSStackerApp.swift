// FITSStackerApp.swift – Application entry point
//
// Native Apple Silicon macOS app built with SwiftUI.
// Requires macOS 14 (Sonoma) and an M-series chip for full Metal acceleration.

import SwiftUI

@main
struct FITSStackerApp: App {

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified(showsTitle: true))
        .windowResizability(.contentMinSize)
        .defaultSize(width: 680, height: 620)
        .commands {
            // Remove New Window shortcut – single-window app
            CommandGroup(replacing: .newItem) {}

            CommandGroup(after: .appInfo) {
                Link("Seestar S50 FITS Guide",
                     destination: URL(string: "https://www.zwoptical.com/seestar-s50")!)
            }
        }
    }
}
