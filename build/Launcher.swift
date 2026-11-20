import Cocoa
import Darwin

final class AppDelegate: NSObject, NSApplicationDelegate {
    var server: Process?
    var stopping = false
    let smoke = ProcessInfo.processInfo.arguments.contains("--smoke-test")
    var smokeChecking = false
    var smokeAttempts = 0
    let support = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/AmuMusicSeparator")

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let item = NSMenuItem()
        menu.addItem(item)
        let appMenu = NSMenu()
        let show = NSMenuItem(title: "打开软件界面", action: #selector(openPage), keyEquivalent: "o")
        show.target = self
        appMenu.addItem(show)
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "退出 Amu 音频分离", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.submenu = appMenu
        NSApp.mainMenu = menu
        do {
            try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
            let resources = Bundle.main.resourceURL!
            let runtime = resources.appendingPathComponent("runtime")
            let application = resources.appendingPathComponent("app")
            let process = Process()
            process.executableURL = runtime.appendingPathComponent("bin/python3")
            process.arguments = ["-B", application.appendingPathComponent("launcher_mac.py").path]
            if smoke { process.arguments!.append("--no-browser") }
            var env = ProcessInfo.processInfo.environment
            env["AMU_VENV"] = runtime.path
            env["AMU_DATA_DIR"] = support.path
            env["AMU_PORT_FILE"] = support.appendingPathComponent("port.txt").path
            env["APPDATA"] = support.deletingLastPathComponent().path
            env["PATH"] = application.appendingPathComponent("bin").path + ":" + runtime.appendingPathComponent("bin").path + ":/usr/bin:/bin:/usr/sbin:/sbin"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["NUMBA_CACHE_DIR"] = support.appendingPathComponent("numba-cache").path
            env["MPLCONFIGDIR"] = support.appendingPathComponent("matplotlib").path
            env["HF_HUB_OFFLINE"] = "1"
            process.environment = env
            process.currentDirectoryURL = support
            let log = support.appendingPathComponent("startup.log")
            FileManager.default.createFile(atPath: log.path, contents: nil)
            let output = try FileHandle(forWritingTo: log)
            process.standardOutput = output
            process.standardError = output
            process.terminationHandler = { [weak self] child in
                DispatchQueue.main.async {
                    guard let self = self, !self.stopping else { return }
                    if self.smoke { self.finishSmoke(false); return }
                    let alert = NSAlert()
                    alert.messageText = "音频分离服务已停止"
                    alert.informativeText = "请重新打开软件。运行记录位于：\(log.path)（退出码 \(child.terminationStatus)）"
                    alert.runModal()
                    NSApp.terminate(nil)
                }
            }
            server = process
            try process.run()
            if smoke {
                Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                    self?.checkSmoke()
                }
            }
        } catch {
            if smoke { finishSmoke(false); return }
            let alert = NSAlert()
            alert.messageText = "无法启动音频分离软件"
            alert.informativeText = error.localizedDescription
            alert.runModal()
            NSApp.terminate(nil)
        }
    }

    func checkSmoke() {
        guard !stopping, !smokeChecking else { return }
        smokeAttempts += 1
        if smokeAttempts > 120 { finishSmoke(false); return }
        guard let text = try? String(contentsOf: support.appendingPathComponent("port.txt"), encoding: .utf8),
              let port = Int(text), let url = URL(string: "http://127.0.0.1:\(port)") else { return }
        smokeChecking = true
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.smokeChecking = false
                if let http = response as? HTTPURLResponse, http.statusCode == 200,
                   let data = data, String(decoding: data, as: UTF8.self).contains("Amu") {
                    self.finishSmoke(true)
                }
            }
        }.resume()
    }

    func finishSmoke(_ passed: Bool) {
        guard !stopping else { return }
        let result: [String: Any] = ["passed": passed, "service_pid": server?.processIdentifier ?? 0,
                                     "data_directory": support.path]
        if let path = ProcessInfo.processInfo.environment["AMU_NATIVE_SMOKE_RESULT"],
           let data = try? JSONSerialization.data(withJSONObject: result, options: .prettyPrinted) {
            try? data.write(to: URL(fileURLWithPath: path))
        }
        NSApp.terminate(nil)
    }

    @objc func openPage() {
        if let port = try? String(contentsOf: support.appendingPathComponent("port.txt"), encoding: .utf8),
           let value = Int(port.trimmingCharacters(in: .whitespacesAndNewlines)),
           (1...65535).contains(value), let url = URL(string: "http://127.0.0.1:\(value)") {
            NSWorkspace.shared.open(url)
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        openPage()
        return true
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopping = true
        if let child = server, child.isRunning {
            // Only our own service's process group; never other local applications.
            if kill(-child.processIdentifier, SIGTERM) != 0 { child.terminate() }
            child.waitUntilExit()
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
