import Foundation
import AppKit
import ApplicationServices

@available(macOS 26.0, *)
@MainActor
final class WindowMigrator: ObservableObject {
    
    @Published private(set) var isWindowMigrated: Bool = false
    @Published private(set) var lastError: String?
    
    private var originalPosition: CGPoint = .zero
    private var targetPID: pid_t = 0
    private var targetWindowElement: AXUIElement?
    
    func moveWindow(pid: pid_t, windowID: CGWindowID, to position: CGPoint) -> Bool {
        guard AXIsProcessTrusted() else {
            lastError = "Accessibility permission required"
            return false
        }
        
        let appElement = AXUIElementCreateApplication(pid)
        
        var windowsRef: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &windowsRef)
        
        guard result == .success, let windows = windowsRef as? [AXUIElement] else {
            lastError = "Failed to get windows: \(result.rawValue)"
            return false
        }
        
        guard let windowElement = windows.first else {
            lastError = "No windows found for PID \(pid)"
            return false
        }
        
        targetPID = pid
        targetWindowElement = windowElement
        
        var positionRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(windowElement, kAXPositionAttribute as CFString, &positionRef) == .success,
           let posValue = positionRef {
            var point = CGPoint.zero
            AXValueGetValue(posValue as! AXValue, .cgPoint, &point)
            originalPosition = point
        }
        
        var newPosition = position
        guard let positionValue = AXValueCreate(.cgPoint, &newPosition) else {
            lastError = "Failed to create position value"
            return false
        }
        
        let setResult = AXUIElementSetAttributeValue(windowElement, kAXPositionAttribute as CFString, positionValue)
        
        if setResult == .success {
            isWindowMigrated = true
            lastError = nil
            return true
        } else {
            lastError = "Failed to set position: \(setResult.rawValue)"
            return false
        }
    }
    
    func moveToVirtualDisplay(pid: pid_t, windowID: CGWindowID, displayID: CGDirectDisplayID) -> Bool {
        guard let screen = getScreen(for: displayID) else {
            lastError = "Virtual display screen not found"
            return false
        }
        
        let targetOrigin = screen.frame.origin
        
        return moveWindow(pid: pid, windowID: windowID, to: targetOrigin)
    }
    
    func restoreWindow() {
        guard isWindowMigrated, let windowElement = targetWindowElement else { return }
        
        var position = originalPosition
        guard let positionValue = AXValueCreate(.cgPoint, &position) else { return }
        
        let result = AXUIElementSetAttributeValue(windowElement, kAXPositionAttribute as CFString, positionValue)
        
        if result != .success {
        }
        
        isWindowMigrated = false
        targetWindowElement = nil
        targetPID = 0
    }
    
    private func getScreen(for displayID: CGDirectDisplayID) -> NSScreen? {
        return NSScreen.screens.first { screen in
            let screenID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID
            return screenID == displayID
        }
    }
}
